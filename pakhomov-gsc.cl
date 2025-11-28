inline ulong rotl(const ulong x, int k) {
    return (x << k) | (x >> (64 - k));
}

inline ulong xoshiro1024_next(ulong state[16], int *p_ptr) {
    int p = *p_ptr;
    int q = p;

    p = (p + 1) & 15;
    ulong s0 = state[p];
    ulong s15 = state[q];

    ulong result = rotl(s0 + s15, 23) + s15;

    s15 ^= s0;
    state[q] = rotl(s0, 25) ^ s15 ^ (s15 << 27);
    state[p] = rotl(s15, 36);

    *p_ptr = p;
    return result;
}

__kernel void seed_search(__global ulong *result, __global ulong *target, ulong bit_count, ulong start_seed, __global int *found_flag) {
    size_t gid = get_global_id(0);
    ulong seed = start_seed + (ulong)gid;

    ulong state[16];
    int p = 0;

    ulong z = seed;
    for (int i = 0; i < 16; ++i) {
        ulong v = z;
        v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9UL;
        v = (v ^ (v >> 27)) * 0x94D049BB133111EBUL;
        v = v ^ (v >> 31);
        state[i] = v;
        z += 0x9E3779B97F4A7C15UL;
    }

    size_t word_count = (bit_count + 63) / 64;
    int match = 1;

    for (size_t i = 0; i < word_count; ++i) {
        ulong generated = xoshiro1024_next(state, &p);

        if (i == word_count - 1 && (bit_count % 64) != 0) {
            const int valid_bits = (int)(bit_count % 64);

            ulong mask;
            if (valid_bits == 0) {
                mask = (ulong)~0ULL;
            } else {
                mask = (( (ulong)1 << (valid_bits) ) - (ulong)1);
                mask = mask << (64 - valid_bits);
            }
            generated &= mask;
        }

        // compare to target word
        if (generated != target[i]) {
            match = 0;
            break;
        }
    }

    if (match) {
        result[0] = seed;
        found_flag[0] = 1;
    }
}
