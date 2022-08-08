#include <cstdlib>
#include <utility>
#include <sstream>
#include <iostream>
#include <string>
#include <assert.h>
#include <cstdio>

#include <bitset>

// C-API
#include "taichi_core_impl.h"
#include "taichi/taichi_core.h"
#include "c_api_test_utils.h"

// GUI
#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>


constexpr int img_c = 4;
constexpr int img_size = 512;
constexpr int init_width = 30720;
constexpr int init_height = 30720;
constexpr int num_runs = 100;
constexpr int max_num_frames = 1000;
constexpr int slice_size = 1000000;
constexpr int num_rows = slice_size / init_width;
constexpr int n = 30720;

template<typename T>
std::vector<T> read_binary_from_file(const std::string& filename,
                                     size_t numel) {
    std::vector<T> result(numel);

    // Open File for Read
    FILE * pFile = fopen(filename.c_str(), "rb"); 
    if (pFile==NULL) {fputs ("File error",stderr); exit (1);}
    
    // Get File Size
    fseek (pFile , 0 , SEEK_END);
    long lSize = ftell (pFile);
    rewind (pFile);
    
    // Read and Check
    int bytes_read = fread(result.data(), sizeof(T), numel, pFile);
    if (bytes_read != lSize) {fputs ("Reading error",stderr); exit (3);}

    fclose(pFile);
    
    return result;
}

void update_sliced_array(TiRuntime runtime,
                         TiMemory memory,
                         void* buffer, 
                         int width,
                         size_t row_index,
                         size_t length,
                         size_t dtype_size) {
    void* ptr = ti_map_memory(runtime, memory);
    char* row_offset = (char*)buffer + row_index * width * dtype_size;
    size_t size_in_bytes = length * dtype_size;
    for(int i = 0; i < length * width; i++) {
        ((char*)ptr)[i] = row_offset[i];
    }

    ti_unmap_memory(runtime, memory);
    
}

static taichi::Arch get_taichi_arch(const std::string& arch_name_) {
    if(arch_name_ == "cuda") {
        return taichi::Arch::cuda;
    }

    if(arch_name_ == "x64") {
        return taichi::Arch::x64;
    }

    TI_ERROR("Unkown arch_name");
    return taichi::Arch::x64;
}
  
static TiArch get_c_api_arch(const std::string& arch_name_) {
    if(arch_name_ == "cuda") {
        return TiArch::TI_ARCH_CUDA;
    }

    if(arch_name_ == "x64") {
        return TiArch::TI_ARCH_X64;
    }
    
    TI_ERROR("Unkown arch_name");
    return TiArch::TI_ARCH_X64;
}
  
static taichi::ui::FieldSource get_field_source(const std::string& arch_name_) {
    if(arch_name_ == "cuda") {
        return taichi::ui::FieldSource::TaichiCuda;
    }

    if(arch_name_ == "x64") {
        return taichi::ui::FieldSource::TaichiX64;
    }
    
    TI_ERROR("Unkown arch_name");
    return taichi::ui::FieldSource::TaichiX64;
}


struct guiHelper {
    std::shared_ptr<taichi::ui::vulkan::Gui> gui_{nullptr};
    std::unique_ptr<taichi::ui::vulkan::Renderer> renderer{nullptr};
    GLFWwindow *window{nullptr};
    taichi::ui::SetImageInfo img_info;

    explicit guiHelper(taichi::lang::DeviceAllocation& devalloc, 
                       const std::string& arch_name) {
      glfwInit();
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
      window = glfwCreateWindow(img_size, img_size, "Taichi show", NULL, NULL);
      if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
      }
    
      // Create a GGUI configuration
      taichi::ui::AppConfig app_config;
      app_config.name = "TaichiSparse";
      app_config.width = img_size;
      app_config.height = img_size;
      app_config.vsync = true;
      app_config.show_window = false;
      app_config.package_path = "."; // make it flexible later
      app_config.ti_arch = get_taichi_arch(arch_name);
      app_config.is_packed_mode = true;

      // Create GUI & renderer
      renderer = std::make_unique<taichi::ui::vulkan::Renderer>();
      renderer->init(nullptr, window, app_config);

      renderer->set_background_color({0.6, 0.6, 0.6});

      gui_ = std::make_shared<taichi::ui::vulkan::Gui>(
          &renderer->app_context(), &renderer->swap_chain(), window);

      // Describe information to render the image
      taichi::ui::FieldInfo f_info;
      f_info.valid = true;
      f_info.field_type = taichi::ui::FieldType::Scalar;
      f_info.matrix_rows = 1;
      f_info.matrix_cols = 1;
      f_info.shape = { img_size, img_size, img_c };

      f_info.field_source = get_field_source(arch_name);
      f_info.dtype = taichi::lang::PrimitiveType::f32;
      f_info.snode = nullptr;
      f_info.dev_alloc = devalloc;
      
      img_info.img = std::move(f_info);
    }

    void step() {
      if (!glfwWindowShouldClose(window)) {
        // Render elements
        renderer->set_image(img_info);
        renderer->draw_frame(gui_.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
      }
    }

    ~guiHelper() {
      gui_.reset();
      renderer.reset();
    }
};

static void taichi_sparse_test(const std::string& arch_name, 
                               const std::string& folder_dir) {
  TiRuntime runtime = ti_create_runtime(get_c_api_arch(arch_name));

  // Load Aot and Kernel
  TiAotModule aot_mod = ti_load_aot_module(runtime, folder_dir.c_str());

  TiKernel k_clear = ti_get_aot_module_kernel(aot_mod, "clear");
  TiKernel k_init_from_slices = ti_get_aot_module_kernel(aot_mod, "init_from_slices");
  TiKernel k_evolve_a_b = ti_get_aot_module_kernel(aot_mod, "evolve_a_b");
  TiKernel k_evolve_b_a = ti_get_aot_module_kernel(aot_mod, "evolve_b_a");
  TiKernel k_fill_img_a = ti_get_aot_module_kernel(aot_mod, "fill_img_a");
  TiKernel k_fill_img_b = ti_get_aot_module_kernel(aot_mod, "fill_img_b");
 
  /* Kernel Initialization */
  const std::vector<int> shape_2d = { num_rows, init_width };
  auto init_buffer_ = capi::utils::make_ndarray(runtime,
                                                TiDataType::TI_DATA_TYPE_U8,
                                                shape_2d.data(), 2,
                                                nullptr, 0,
                                                false /*host_read*/, false /*host_write*/
                                                );
  TiArgument init_buffer_args[4];
  init_buffer_args[0] = init_buffer_.arg_;

  TiArgument init_width_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                               .value = {.i32 = init_width}};
  init_buffer_args[1] = init_width_arg;
  
  // Read init_buffer from binary file -> init_buffer_val
  std::string binary_filename = folder_dir + "/init_buffer.bin";
  std::vector<char> init_buffer_val = read_binary_from_file<char>(binary_filename, init_width * init_height);

  // Initialize with slice
  ti_launch_kernel(runtime, k_clear, 0, &init_buffer_args[0]);
  for(int i = 0; i < init_height; i = i + num_rows) {
        // i
        TiArgument i_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                            .value = {.i32 = i}};
        init_buffer_args[2] = i_arg;
        
        // rows
        int rows = std::min(num_rows, init_height - i);
        TiArgument rows_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                               .value = {.i32 = rows}};
        init_buffer_args[3] = rows_arg;
        
        // init_buffer[i:i+num_rows, :] -> init_buffer_
        update_sliced_array(runtime, init_buffer_.memory_,
                            init_buffer_val.data()/* void* buffer */,
                            init_width/* int width */,
                            i/* int row_index */,
                            num_rows /* length */,
                            1 /* size_t dtype_size */);
        
        ti_launch_kernel(runtime, k_init_from_slices, 4, &init_buffer_args[0]);
  }

  /* Prepare Image for GUI*/
  TiArgument region_size_arg = {.type = TiArgumentType::TI_ARGUMENT_TYPE_I32,
                                .value = {.i32 = n}};
  
  const std::vector<int> shape_3d = { img_size, img_size, img_c };
  auto arr_ = capi::utils::make_ndarray(runtime,
                                        TiDataType::TI_DATA_TYPE_F32,
                                        shape_3d.data(), 3,
                                        nullptr, 0,
                                        false /*host_read*/, false /*host_write*/
                                        );
  TiArgument arr_args[2] = { region_size_arg, arr_.arg_ };

  Runtime* real_runtime = (Runtime *)runtime;
  taichi::lang::DeviceAllocation devalloc = devmem2devalloc(*real_runtime, arr_.memory_);
  guiHelper gui_helper(devalloc, arch_name);

  for(size_t frame = 0; frame < max_num_frames; frame++) {

      // running(state_a, state_b)
      for(size_t i = 0; i < num_runs; i++) {
        ti_launch_kernel(runtime, k_evolve_a_b, 0, &arr_args[0]);
        ti_launch_kernel(runtime, k_evolve_b_a, 0, &arr_args[0]);
      }
      
      ti_launch_kernel(runtime, k_fill_img_a, 2, &arr_args[0]);
      ti_wait(runtime);

      gui_helper.step();

      // running(state_b, state_a)
      for(size_t i = 0; i < num_runs; i++) {
        ti_launch_kernel(runtime, k_evolve_b_a, 0, &arr_args[0]);
        ti_launch_kernel(runtime, k_evolve_a_b, 0, &arr_args[0]);
      }
      ti_launch_kernel(runtime, k_fill_img_b, 2, &arr_args[0]);
      ti_wait(runtime);
      
      gui_helper.step();
  }
  

  ti_destroy_aot_module(aot_mod);
  ti_destroy_runtime(runtime);
}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    std::string folder_dir = argv[1];
    std::string arch_name = argv[2];

    taichi_sparse_test(arch_name, folder_dir);

    return 0;
}
