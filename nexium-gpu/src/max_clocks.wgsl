struct Load {
    rounds: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> sink: array<u32>;
var<immediate> load: Load;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let index = (group.y * groups.x + group.x) * 64u + local;
    var state = (index ^ load.seed) * 0x9e3779b9u + 1u;
    for (var i = 0u; i < load.rounds; i = i + 1u) {
        state = state ^ (state << 13u);
        state = state ^ (state >> 17u);
        state = state ^ (state << 5u);
        state = state * 0x2c1b3c6du + i;
    }
    let slot = index % arrayLength(&sink);
    sink[slot] = sink[slot] ^ state;
}
