import argparse
import os
from taichi.lang.impl import grouped
import taichi as ti
import numpy as np

def from_rle(file):
    width = -1
    height = -1
    data = np.zeros((4, 4), dtype=np.uint8)
    x, y, n = 0, 0, 0
    count = 0
    with open(file, "r") as f:
        for line in f:
            if line[0] == '#':
                # skip comment lines
                continue
            elif line[0] == 'x':
                # read params
                tokens = line.split(',')
                width = int(tokens[0][4:])
                height = int(tokens[1][4:])
                data = np.zeros((height, width), dtype=np.uint8)
            else:
                # process RLE-encoded pattern
                for c in line:
                    if c.isdigit():
                        n = n * 10 + int(c)
                    else:
                        if n == 0:
                            n = 1
                        if c == 'b':
                            for j in range(x, x + n):
                                data[y][j] = 0
                            x += n
                        elif c == 'o':
                            for j in range(x, x + n):
                                data[y][j] = 1
                            x += n
                        elif c == '$':
                            x = 0
                            y += n
                        elif c == '!':
                            break
                        n = 0
    return data, width, height


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--arch", type=str)
args = parser.parse_args()

if args.arch == "cuda":
    arch = ti.cuda
elif args.arch == "x64":
    arch = ti.x64
else:
    assert False

ti.init(arch=arch)

img_size = 512
img_c = 4
N = 65536
bits = 32
n_blocks = 16
n = 30720
boundary_offset = int((N - n) / 2)

qu1 = ti.types.quant.int(1, False)

state_a = ti.field(dtype=qu1)
state_b = ti.field(dtype=qu1)
block = ti.root.pointer(ti.ij, (n_blocks, n_blocks))
block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(
    ti.j, bits, max_num_bits=bits).place(state_a)
block.dense(ti.ij, (N // n_blocks, N // (bits * n_blocks))).quant_array(
    ti.j, bits, max_num_bits=bits).place(state_b)

arr = ti.ndarray(ti.f32, shape=(img_size, img_size, img_c))


@ti.kernel
def evolve_a_b():
    ti.lang.impl.get_runtime().prog.current_ast_builder().bit_vectorize()
    for i, j in state_a:
        num_active_neighbors = ti.u32(0)
        num_active_neighbors += ti.cast(state_a[i - 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_a[i - 1, j], ti.u32)
        num_active_neighbors += ti.cast(state_a[i - 1, j + 1], ti.u32)
        num_active_neighbors += ti.cast(state_a[i, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_a[i, j + 1], ti.u32)
        num_active_neighbors += ti.cast(state_a[i + 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_a[i + 1, j], ti.u32)
        num_active_neighbors += ti.cast(state_a[i + 1, j + 1], ti.u32)
        state_b[i, j] = (num_active_neighbors == 3) | ((num_active_neighbors == 2) & (state_a[i, j] == 1))

@ti.kernel
def evolve_b_a():
    ti.lang.impl.get_runtime().prog.current_ast_builder().bit_vectorize()
    for i, j in state_b:
        num_active_neighbors = ti.u32(0)
        num_active_neighbors += ti.cast(state_b[i - 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_b[i - 1, j], ti.u32)
        num_active_neighbors += ti.cast(state_b[i - 1, j + 1], ti.u32)
        num_active_neighbors += ti.cast(state_b[i, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_b[i, j + 1], ti.u32)
        num_active_neighbors += ti.cast(state_b[i + 1, j - 1], ti.u32)
        num_active_neighbors += ti.cast(state_b[i + 1, j], ti.u32)
        num_active_neighbors += ti.cast(state_b[i + 1, j + 1], ti.u32)
        state_a[i, j] = (num_active_neighbors == 3) | ((num_active_neighbors == 2) & (state_b[i, j] == 1))


@ti.func
def fill_pixel(scale, buffer, i, j, region_size):
    ii = i * 1.0 / img_size
    jj = j * 1.0 / img_size
    ret_val = 0.0
    if scale > 1:
        sx1, sx2, sy1, sy2 = j * scale, (j + 1) * scale, i * scale, (i + 1) * scale
        x1 = ti.cast(sx1, ti.i32)
        x2 = ti.cast(sx2, ti.i32) + 1
        y1 = ti.cast(sy1, ti.i32)
        y2 = ti.cast(sy2, ti.i32) + 1
        count = 0
        val = 0
        for mm in range(y1, y2):
            for nn in range(x1, x2):
                if mm + 0.5 > sy1 and mm + 0.5 < sy2 and nn + 0.5 > sx1 and nn + 0.5 < sx2:
                    count += 1
                    val += buffer[boundary_offset + int(n / 2) - int(region_size / 2) + mm,
                                  boundary_offset + int(n / 2) - int(region_size / 2) + nn]
        ret_val = val
    else:
        ret_val = buffer[int(boundary_offset + int(n / 2) - region_size / 2 + region_size * ii),
                   int(boundary_offset + int(n / 2) - region_size / 2  + region_size * jj)]
    return ret_val


@ti.kernel
def fill_img_a(region_size: ti.i32, arr: ti.types.ndarray()):
    scale = region_size * 1.0 / img_size
    for i, j in ti.ndrange(img_size, img_size):
        for c in range(img_c):
            arr[i, j, c] = fill_pixel(scale, state_a, i, j, region_size)

@ti.kernel
def fill_img_b(region_size: ti.i32, arr: ti.types.ndarray()):
    scale = region_size * 1.0 / img_size
    for i, j in ti.ndrange(img_size, img_size):
        for c in range(img_c):
            arr[i, j, c] = fill_pixel(scale, state_b, i, j, region_size)

@ti.kernel
def clear():
    for i in range(boundary_offset, N - boundary_offset):
        for j in range(boundary_offset, N - boundary_offset):
            state_a[i, j] = 0
            state_b[i, j] = 0

@ti.kernel
def init_from_slices(init_buffer: ti.types.ndarray(),
                     init_width: ti.i32, offset: ti.i32, rows: ti.i32):
    for i in range(boundary_offset + offset, boundary_offset + offset + rows):
        for j in range(boundary_offset, boundary_offset + init_width):
            state_a[i, j] = init_buffer[i - boundary_offset - offset, j - boundary_offset]

def save_kernels(arch, dirname):
    init_buffer, init_width, init_height = from_rle('metapixel-galaxy.rle')
    init_buffer = np.rot90(init_buffer, 3)
    
    print(init_buffer.shape)
    print(init_buffer.dtype)

    binary_filename = os.path.join(dirname, "init_buffer.bin")
    init_buffer.tofile(binary_filename)

    m = ti.aot.Module(arch)
    
    # init
    m.add_kernel(clear, template_args={})
    m.add_kernel(init_from_slices, template_args={'init_buffer': init_buffer})

    # run
    m.add_kernel(evolve_a_b, template_args={})
    m.add_kernel(evolve_b_a, template_args={})
    m.add_kernel(fill_img_a, template_args={'arr': arr})
    m.add_kernel(fill_img_b, template_args={'arr': arr})
    
    m.add_field("state_a", state_a)
    m.add_field("state_b", state_b)

    m.save(dirname, 'whatever')

if __name__ == '__main__':
    assert args.dir is not None
    save_kernels(arch=arch, dirname=args.dir)
