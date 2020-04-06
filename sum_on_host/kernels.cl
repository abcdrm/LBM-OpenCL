#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define NSPEEDS         9
#define CALC_IDX(i, j, nx, ny, speed) ((nx * ny * speed) + ((j * nx) + i))
#define c_sq            (1.f / 3.f)
#define c_2sq2          (2.f * c_sq * c_sq)
#define c_2sq           (2.f * c_sq)
#define w0              (4.f / 9.f)
#define w1              (1.f / 9.f)
#define w2              (1.f / 36.f)
#define BLOCK_SIZE      32

void reduce(
    local float*,
    global float*,
    int,
    int);

inline void atomic_float_add(volatile __global float* addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

kernel void accelerate_flow(global float* cells,
    global int* obstacles,
    int nx, int ny,
    float density, float accel)
{
    /* compute weighting factors */
    float w1f = density * accel / 9.f;
    float w2f = density * accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj = ny - 2;

    /* get column index */
    int ii = get_global_id(0);

    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj * nx]
        && (cells[CALC_IDX(ii, jj, nx, ny, 3)] - w1f) > 0.f
        && (cells[CALC_IDX(ii, jj, nx, ny, 6)] - w2f) > 0.f
        && (cells[CALC_IDX(ii, jj, nx, ny, 7)] - w2f) > 0.f)
    {
        /* increase 'east-side' densities */
        cells[CALC_IDX(ii, jj, nx, ny, 1)] += w1f;
        cells[CALC_IDX(ii, jj, nx, ny, 5)] += w2f;
        cells[CALC_IDX(ii, jj, nx, ny, 8)] += w2f;
        /* decrease 'west-side' densities */
        cells[CALC_IDX(ii, jj, nx, ny, 3)] -= w1f;
        cells[CALC_IDX(ii, jj, nx, ny, 6)] -= w2f;
        cells[CALC_IDX(ii, jj, nx, ny, 7)] -= w2f;
    }
}

kernel void propagate(global float* cells,
    global float* tmp_cells,
    global int* obstacles,
    int nx, int ny, float omega,
    local float* local_av_vels, int tt,
    global float* global_av_vels, 
    int tot_cells)
{
    /* get column and row indices */
    int ii = get_global_id(0);
    int jj = get_global_id(1);
    int local_ii = get_local_id(0);
    int local_jj = get_local_id(1);

    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    float speeds[NSPEEDS];
    speeds[0] = cells[CALC_IDX(ii, jj, nx, ny, 0)]; /* central cell, no movement */
    speeds[1] = cells[CALC_IDX(x_w, jj, nx, ny, 1)]; /* east */
    speeds[2] = cells[CALC_IDX(ii, y_s, nx, ny, 2)]; /* north */
    speeds[3] = cells[CALC_IDX(x_e, jj, nx, ny, 3)]; /* west */
    speeds[4] = cells[CALC_IDX(ii, y_n, nx, ny, 4)]; /* south */
    speeds[5] = cells[CALC_IDX(x_w, y_s, nx, ny, 5)]; /* north-east */
    speeds[6] = cells[CALC_IDX(x_e, y_s, nx, ny, 6)]; /* north-west */
    speeds[7] = cells[CALC_IDX(x_e, y_n, nx, ny, 7)]; /* south-west */
    speeds[8] = cells[CALC_IDX(x_w, y_n, nx, ny, 8)]; /* south-east */

    /* compute local density total */
    float local_density;
    local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
    /* compute x velocity component */
    float u_x = (speeds[1] + speeds[5] + speeds[8] - speeds[3] - speeds[6] - speeds[7]) / local_density;
    /* compute y velocity component */
    float u_y = (speeds[2] + speeds[5] + speeds[6] - speeds[4] - speeds[7] - speeds[8]) / local_density;

    if (obstacles[ii + jj * nx]) {
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 0)] = speeds[0];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 1)] = speeds[3];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 2)] = speeds[4];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 3)] = speeds[1];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 4)] = speeds[2];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 5)] = speeds[7];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 6)] = speeds[8];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 7)] = speeds[5];
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 8)] = speeds[6];
    }
    else {
        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u1 = u_x;        /* east */
        float u2 = u_y;  /* north */
        float u3 = -u_x;        /* west */
        float u4 = -u_y;  /* south */
        float u5 = u_x + u_y;  /* north-east */
        float u6 = -u_x + u_y;  /* north-west */
        float u7 = -u_x - u_y;  /* south-west */
        float u8 = u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f - u_sq / (c_2sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u1 / c_sq + (u1 * u1) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[2] = w1 * local_density * (1.f + u2 / c_sq + (u2 * u2) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[3] = w1 * local_density * (1.f + u3 / c_sq + (u3 * u3) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[4] = w1 * local_density * (1.f + u4 / c_sq + (u4 * u4) / (c_2sq2) - u_sq / (c_2sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u5 / c_sq + (u5 * u5) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[6] = w2 * local_density * (1.f + u6 / c_sq + (u6 * u6) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[7] = w2 * local_density * (1.f + u7 / c_sq + (u7 * u7) / (c_2sq2) - u_sq / (c_2sq));
        d_equ[8] = w2 * local_density * (1.f + u8 / c_sq + (u8 * u8) / (c_2sq2) - u_sq / (c_2sq));

        /* relaxation step */
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 0)] = speeds[0] + omega * (d_equ[0] - speeds[0]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 1)] = speeds[1] + omega * (d_equ[1] - speeds[1]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 2)] = speeds[2] + omega * (d_equ[2] - speeds[2]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 3)] = speeds[3] + omega * (d_equ[3] - speeds[3]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 4)] = speeds[4] + omega * (d_equ[4] - speeds[4]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 5)] = speeds[5] + omega * (d_equ[5] - speeds[5]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 6)] = speeds[6] + omega * (d_equ[6] - speeds[6]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 7)] = speeds[7] + omega * (d_equ[7] - speeds[7]);
        tmp_cells[CALC_IDX(ii, jj, nx, ny, 8)] = speeds[8] + omega * (d_equ[8] - speeds[8]);
    }

    local_av_vels[local_ii + local_jj * BLOCK_SIZE] = (obstacles[ii + jj * nx]) ? 0 : (native_sqrt((u_x * u_x) + (u_y * u_y)) / tot_cells);
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce(local_av_vels, global_av_vels, tt, tot_cells);
}

void reduce(local float* local_av_vels, 
    global float* global_av_vels, 
    int tt, int tot_cells) 
{
    int local_ii = get_local_id(0);
    int local_jj = get_local_id(1);

    float sum;
    int i;

    if (local_ii == 0 && local_jj == 0) {
        sum = 0.0f;

        for (i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
            sum += local_av_vels[i];
        }

        global_av_vels[get_group_id(0) + get_group_id(1) * get_num_groups(0)] = sum;
    }
}