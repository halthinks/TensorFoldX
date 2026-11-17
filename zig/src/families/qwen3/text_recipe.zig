//! A Qwen3 text checkpoint (Qwen3ForCausalLM, BF16) as a cuda_dense recipe: QK norm and RoPE through the shared qk rows.
const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense");
const cfg = @import("config.zig");
const wts = @import("text_weights.zig");

const Recipe = @This();
const P = *const dense.Pass(Recipe);

pub const name = "Qwen3";
pub const model_type = "qwen3";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 32768;

pub const Weights = wts.Weights;

/// The recipe's own launches: the shared QK norm and RoPE rows.
pub const Kernels = struct {
    qk: cuda.qk_rows.QkRows,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        return .{ .qk = try cuda.qk_rows.QkRows.load(ctx.d) };
    }

    pub fn deinit(k: *Kernels) void {
        k.qk.unload();
    }
};

/// The engine's geometry of an admitted Qwen3 config.
pub fn geometryOf(c: cfg.Spec) !dense.Geometry {
    if (c.head_dim != 128) return error.UnsupportedQwen3Text;
    const kvw = c.kv_heads * c.head_dim;
    return .{ .hidden = c.hidden, .layers = c.layers, .heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .vocab = c.vocab, .eps = @floatCast(c.eps), .context = c.max_positions, .acc = c.heads * c.head_dim + 2 * kvw, .wide = 2 * c.inter, .mid = c.inter, .theta = c.theta, .inv_freq = if (c.theta == 1e6) &hf_inv_freq else null };
}

/// Transformers' own float32 inv_freq for theta 1e6 at head dim 128 (as dots.mocr's fixture: Torch's CPU pow, bit for bit).
const hf_inv_freq: [64]f32 = blk: {
    @setEvalBranchQuota(20_000);
    const text = @embedFile("fixtures/rope_inv_freq.txt");
    var out: [64]f32 = undefined;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    for (&out) |*v| v.* = @bitCast(std.fmt.parseInt(u32, it.next().?, 0) catch unreachable);
    break :blk out;
};

pub fn geometry(a: std.mem.Allocator, io: std.Io, dir: []const u8) !dense.Geometry {
    return geometryOf(try wts.readSpec(a, io, dir));
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !Weights {
    return wts.load(gpa, io, d, s, dir);
}

pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    return wts.textBytes(try wts.readSpec(a, io, dir));
}

pub fn embed(p: P) !void {
    try p.k.embed.bf16(p.s, p.m.s.ids, p.w.embed, p.m.s.h, p.rows, p.g.hidden);
}

/// One Qwen3 layer: RMS, q|k|v rounded to bf16, each q and k head's RMS norm then RoPE, attention, o, RMS, SwiGLU.
pub fn layer(p: P, i: usize) !void {
    const x = p.m.s;
    const l = p.w.layers[i];
    const kvw = p.g.kvWidth();
    try p.norm(if (i == 0) null else x.r, l.ln1);
    // the fp32 scratch holds the bf16 q|k|v rows: half its bytes
    try p.project(l.qkv, x.x, x.acc);
    const heads: cuda.qk_rows.Heads = .{ .h = p.g.heads, .hk = p.g.kv_heads, .dim = p.g.head_dim };
    const shape: cuda.qk_rows.Shape = .{ .norms = l.qk_norm, .norms_bf16 = true, .inv = p.m.inv, .rot = p.g.head_dim, .eps = p.g.eps, .round = true };
    // keys and values a row side by side, as the engine's cache write reads them
    const out: cuda.qk_rows.Out = .{ .q = x.q, .k = x.kv, .v = x.kv + kvw * 2, .kv_ld = 2 * kvw };
    try p.own.qk.rows(p.s, x.acc, heads, shape, x.pos, null, out, p.rows);
    try p.attend(i, x.q, x.kv, x.y);
    try p.project(l.o, x.y, x.r);
    try p.norm(x.r, l.ln2);
    try p.project(l.gate_up, x.x, x.wide);
    try p.siluMul(x.wide, x.y, p.g.mid);
    try p.project(l.down, x.y, x.r);
}

pub fn final(p: P) !void {
    try p.norm(p.m.s.r, p.w.norm);
}

pub fn head(w: *const Weights) cuda.qmmf.Weight {
    return w.head;
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.InvalidConfig, error.UnsupportedConfig, error.UnsupportedQwen3Text => "this checkpoint's config is not a Qwen3 text model this engine serves (Qwen3ForCausalLM in BF16, head_dim 128, no attention bias, plain RoPE)",
        error.Qwen3TensorCount => "this Qwen3 checkpoint has tensors the text engine does not read; serve the released BF16 files",
        else => null,
    };
}

test "Qwen3-8B's geometry: 32 query heads over 8 KV heads of 128, q|k|v 6144 wide" {
    const s: cfg.Spec = .{ .layers = 36, .hidden = 4096, .heads = 32, .kv_heads = 8, .head_dim = 128, .inter = 12288, .vocab = 151936, .eps = 1e-6, .theta = 1e6, .max_positions = 40960, .tied = false };
    const g = try geometryOf(s);
    try std.testing.expect(g.acc == 6144 and g.wide == 24576 and g.mid == 12288 and g.qWidth() == 4096 and g.kvWidth() == 1024 and g.group() == 4);
    var odd = s;
    odd.head_dim = 64;
    try std.testing.expectError(error.UnsupportedQwen3Text, geometryOf(odd));
}
