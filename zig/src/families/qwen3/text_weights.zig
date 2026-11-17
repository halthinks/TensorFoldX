//! A Qwen3 text checkpoint on CUDA for the dense engine: BF16 tensors on the device, q|k|v and gate|up fused into lane weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Weight = cuda.qmmf.Weight;

pub const Layer = struct {
    ln1: u64, // bf16 [hidden]: input_layernorm
    ln2: u64, // bf16 [hidden]: post_attention_layernorm
    qkv: Weight, // q|k|v [hidden + 2 kv width, hidden], no bias
    qk_norm: u64, // bf16 [2, head_dim]: q_norm, then k_norm
    o: Weight,
    gate_up: Weight, // gate|up [2 intermediate, hidden]
    down: Weight,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    spec: cfg.Spec,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    layers: []Layer = &.{},
    embed: u64 = 0, // bf16 [vocab, hidden]
    norm: u64 = 0, // bf16 [hidden]: model.norm
    head: Weight = undefined, // lm_head, or the embedding table again when the config ties them

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !DeviceBuffer {
        try w.buffers.ensureUnusedCapacity(w.gpa, 1);
        const b = try DeviceBuffer.alloc(d, bytes);
        w.buffers.appendAssumeCapacity(b);
        return b;
    }
};

/// The spec of the config.json in `dir`.
pub fn readSpec(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !cfg.Spec {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return cfg.parse(gpa, bytes);
}

/// The tensors' bytes on the device: every projection as stored (their rows are multiples of 128), norms, both tables.
pub fn textBytes(s: cfg.Spec) u64 {
    const h: u64 = s.hidden;
    const kvw: u64 = s.kv_heads * s.head_dim;
    const qw: u64 = s.heads * s.head_dim;
    const m: u64 = s.inter;
    const layer = 2 * h + (qw + 2 * kvw) * h + 2 * s.head_dim + h * qw + 3 * m * h;
    const tables: u64 = if (s.tied) 1 else 2;
    return (tables * s.vocab * h + h + s.layers * layer) * 2;
}

const Loader = struct {
    w: *Weights,
    ck: *core.Checkpoint,
    d: *const cuda.Driver,
    s: cuda.Stream,
    pack: *const cuda.modelopt.Packer,
    stage: DeviceBuffer,

    /// bf16 tensors [rows[i], k] one after another in a buffer of their own.
    fn joined(l: *Loader, names: []const []const u8, rows: []const usize, k: usize) !u64 {
        var n: usize = 0;
        for (rows) |r| n += r;
        const b = try l.w.alloc(l.d, n * k * 2);
        var at: usize = 0;
        for (names, rows) |name, r| {
            const t = try l.ck.expect(name, .bf16, if (k == 1) &.{r} else &.{ r, k });
            try b.upload(at, t.bytes);
            at += t.bytes.len;
        }
        return b.ptr;
    }

    /// bf16 projections [rows[i], k] stacked into one [sum rows, k] lane weight, packed on the device.
    fn lane(l: *Loader, names: []const []const u8, rows: []const usize, k: usize) !Weight {
        var n: usize = 0;
        var at: usize = 0;
        for (names, rows) |name, r| {
            const t = try l.ck.expect(name, .bf16, &.{ r, k });
            try l.stage.upload(at, t.bytes);
            at += t.bytes.len;
            n += r;
        }
        const b = try l.w.alloc(l.d, cuda.modelopt.padded(n) * k * 2);
        try l.pack.bf16Lane(l.s, l.stage.ptr, b.ptr, n, k);
        try l.s.synchronize(); // the next projection's upload reuses the stage
        return cuda.modelopt.bf16Weight(b.ptr, n, k);
    }

    fn settle(l: *const Loader) void {
        cuda.completion.drain(l.d, l.s, "a Qwen3 text weight load did not drain before its buffers were freed");
        cuda.completion.drain(l.d, .{ .d = l.d, .handle = null }, "Qwen3 text weight uploads did not drain before their buffers were freed");
    }
};

/// The checkpoint at `dir` on the device; a tensor missing, of another shape or left over refuses.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !Weights {
    const c = try readSpec(gpa, io, dir);
    var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &.{ "model.", "lm_head." });
    defer ck.close();
    var pack = try cuda.modelopt.Packer.load(d);
    defer pack.unload();
    var w: Weights = .{ .gpa = gpa, .spec = c };
    errdefer w.deinit();
    w.layers = try gpa.alloc(Layer, c.layers);
    const h = c.hidden;
    const hd = c.head_dim;
    const qw = c.heads * hd;
    const kvw = c.kv_heads * hd;
    const m = c.inter;
    var stage = try DeviceBuffer.alloc(d, @max(c.vocab, qw + 2 * kvw, 2 * m) * @max(h, m) * 2);
    defer stage.free();
    var l: Loader = .{ .w = &w, .ck = &ck, .d = d, .s = s, .pack = &pack, .stage = stage };
    errdefer l.settle();
    w.embed = try l.joined(&.{"model.embed_tokens.weight"}, &.{c.vocab}, h);
    w.norm = try l.joined(&.{"model.norm.weight"}, &.{h}, 1);
    w.head = try l.lane(&.{if (c.tied) "model.embed_tokens.weight" else "lm_head.weight"}, &.{c.vocab}, h);
    const parts = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm" };
    var names: [parts.len][96]u8 = undefined;
    for (w.layers, 0..) |*y, i| {
        var n: [parts.len][]const u8 = undefined;
        for (&n, &names, parts) |*out, *buf, part| out.* = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}.weight", .{ i, part });
        y.* = .{
            .ln1 = try l.joined(n[9..10], &.{h}, 1),
            .ln2 = try l.joined(n[10..11], &.{h}, 1),
            .qkv = try l.lane(n[0..3], &.{ qw, kvw, kvw }, h),
            .qk_norm = try l.joined(n[7..9], &.{ hd, hd }, 1),
            .o = try l.lane(n[3..4], &.{h}, qw),
            .gate_up = try l.lane(n[4..6], &.{ m, m }, h),
            .down = try l.lane(n[6..7], &.{h}, m),
        };
    }
    if (ck.unused() != 0) return error.Qwen3TensorCount;
    try s.synchronize();
    return w;
}

test "Qwen3-8B's text bytes: two tables, the final norm and 36 layers with their QK norms" {
    const s: cfg.Spec = .{ .layers = 36, .hidden = 4096, .heads = 32, .kv_heads = 8, .head_dim = 128, .inter = 12288, .vocab = 151936, .eps = 1e-6, .theta = 1e6, .max_positions = 40960, .tied = false };
    try std.testing.expectEqual(@as(u64, 16381470720), textBytes(s));
}
