//! Qwen3 text checkpoints on CUDA: the Qwen3 recipe behind the shared dense engine.
const dense = @import("cuda_dense");

pub const recipe = @import("text_recipe.zig");
pub const weights = @import("text_weights.zig");
pub const native = dense.Native(recipe);

test {
    _ = recipe;
    _ = weights;
}
