<!-- toc start: 4 [do not erase this comment] -->
**Table of contents**
- [CUDA Graphic Engine (CGE)](#cuda-graphic-engine-cge)
	- [Introduction](#introduction)
	- [Build and run](#build-and-run)
	- [Framework and libraries](#framework-and-libraries)
	- [Input Data](#input-data)
		- [Model files](#model-files)
		- [Textures](#textures)
	- [Pipeline](#pipeline)
	- [CUDA optimizations](#cuda-optimizations)
		- [Points viewing](#points-viewing)
		- [Other pipeline stages](#other-pipeline-stages)
			- [Rasterization](#rasterization)
	- [Performance comparisons](#performance-comparisons)
	- [Future updates](#future-updates)
<!-- toc end [do not erase this comment] -->

# CUDA Graphic Engine (CGE)

## Introduction
The project aims to create a graphic engine based on CUDA able of rendering 3D objects on the screen, using perspective projection to tranform 3D points into two dimensions. The application can also fit images (called **textures**) on the surfaces of the objects, so it is possible to represent different materials. Everything is showned from the perspective of a camera that can be moved and rotated through the mouse and the keyboard.

## Build and run
```bash
nvcc sources/*.cu -o cge -lcublas -lSDL2 -lSDL2_image -arch=sm_xx -Xcompiler -DDEVICE=1
./cge
```

The `-DDEVICE` option let you choose whether the engine should use the GPU or not. If it is set to `0`, GPU acceleration is disabled. Set it to `1` for enabling it.

> The engine only works on GPUs with SM 6.0 or higher.

## Framework and libraries
The engine is totally written in **C++**, using the **CUDA** framework to optimize the computations through parallelization on the GPU. The **cuBLAS** library is also used to simplify matrix operations, while the **SDL** library is responsible for managing the graphic and the inputs coming from the keyboard and the mouse.

## Input Data
In order to visualize a 3D object on the screen, the engine needs some informations about its structure and composition. Those data are stored in a **mesh**, which represents the 3D object through four main informations:
- The relative positions of all the 3D points of the object, stored into a matrix of type `double`;
- The **UV coordinates** for texture rastering, stored into a matrix of type `double`. They consist of 2D vectors taking values between $0$ and $1$, which correspond to specific points of the texture, that can be whatever image of any dimension.
- **Point faces** information: the 3D object consists of a bunch of triangles in the 3D space, described by their three vertices. So the model must specify how the points are connected to each other in order to construct the lattice of triangles that shapes the object. These data are stored into a matrix of `int` types corresponding to the indices of the points which make the triangles.
- **Texture UV faces** is an index matrix used in rasterization which associate face points with texture UV coordinates. Indeed, this matrix has the same dimensions and contains the same value types of the face points matrix.

### Model files
Model information are usually stored into structured files that follow a specific standard and have the `.obj` extension. These files contain all the informations that the engine needs in order to show that mesh on the screen. The information are stored line by line, where the starting word on each line indicates which kind of information it contains:
- Lines starting with the **`v`** character provide the three coordinates of a point, each separated by a whitespace.
- Lines starting with **`vt`** contains one texture point, described by its UV coordinates separated by a whitespace.
- Lines starting with the **`f`** character indicates a *face* of the object. It usually consists of a list of 3 or 4 positive integers separated by a whitespace, which stands for the indices of the points involved in that face. In order to associate each face point to a specific point of the texture (so that the texture can be correctly fitted on it), each index is flanked with the index of the related texture point, separated by a `/` character, for example: `f 1/1 2/2 3/3`. Since the engine only works with triangular faces, quadrilaters must be subdivided into two triangles.

### Textures
Texture are nothing else than square images. The more large are their dimensions, the more sharp will be the graphic of the engine, at the expense of the performances.

## Pipeline
The term ***pipeline*** refers to the whole process of transforming the points and perform computations up to the rendering of meshes on the screen. Let's see all of its stages more in detail:
1. **Points viewing**: on this stage, the points of the mesh currently processed are firstly rotated according to the rotation angles of the mesh. This allows the mesh to be rotated on itself, but only if the model refers to a cartesian system having its origin on the center point of the object. Then the world position of the object is added to each one of its points, while the camera position is subtracted from them. This is because of the illusion of moving around the world which involves the object to always move in the opposite direction of the camera. The same thing happens for the rotation of the object according to the opposite of the camera rotation angles.
1. **Normals computations**: normals are unit vectors which indicates the facing of each triangle of the mesh. They can be used to apply shaders on the mesh. The **shader** indicates how a mesh reflects and diffuses the surrounding lights. To compute normals, the engine performs the cross product between two sides of each triangle, then it sums the computed vector to the normals corresponding to each point of the triangle. This way, if we use a shader, the transition between one face and the other is much smoother.
1. **Face decomposition**: on this stage, the processed points are inserted into a list of vertices which follows the connections between the points, in order to construct the faces as specified by the model, under the shape of triangles. Face decomposition also regards other data of the mesh, so to each face vertex is associated the corresponding normal and the texture point. During this stage, we also compute faces normals in order to check for visible faces and discard the invisible ones.
1. **Frustum culling**: it is needed in order to limit the rendering to a small portion of the world, which corresponds to that captured by the camera. This avoids an overload of the engine when the world gets too big. To do that the engine generates a **frustum**, that is a set of planes in the 3D space, described by their positions and normals, which delimit the portion of world captured by the camera. Triangles outside those planes are discarded, those inside are keeped, while those who cross frustum planes are clipped into one or two smaller triangles.
1. **Perspective projection**: in this stage each 3D point in projected into the 2D space by following these formulas:
$$x_{proj} = \frac{x}{|z|}\cdot s+\frac{w}{2}$$
$$y_{proj} = \frac{-y}{|z|}\cdot s+\frac{h}{2}$$

where:
- $x_{proj}$ is the projected $x$ coordinate
- $y_{proj}$ is the projected $y$ coordinate
- $x$ is the processed $x$ coordinate of the current point
- $y$ is the processed $y$ coordinate of the current point
- $z$ is the processed $z$ coordinate of the current point
- $s$ is the *scale* of the screen
- $w$ is the width of the screen
- $h$ is the height of the screen

> Note that points in the origin will be projected on the center of the screen.

For performance improvement purposes, the pipeline also computes another information, that is the *rendering area*. Since each mesh occupies a small portion of the screen on the average, it is clever to scan only the pixels which are inside the minimum rectangle that contains all the faces of the mesh instead of the whole screen. This can be done by looping through all the projected points of the mesh and search for the top left and the bottom right corners of the rectangle.

## CUDA optimizations
In order to reduce the load of work that the CPU should face, the engine uses the CUDA framework as the interface for the GPU so that it can use parallelization to accelerate the computation of the heaviest jobs. Let's see where and how CUDA is used.

### Points viewing
In the stage of points viewing, points must be rotated and translated. For the rotation part, we have to multiply the point matrix with the rotation matrix, so we can use the cuBLAS library and the `cublasDgemm` method to perform this computation. The translation part, simply consists of a summation of the shift vector to the rows of the point matrix, so we can build a kernel which parallelizes this computation on the rows of the matrix. The same thing can be done for the subtraction of the camera position vector from the rows of the point matrix.

### Other pipeline stages
All other stages of the pipeline use CUDA to accelerate the computation through kernels which parallelize the work on the first dimension of the considered matrix. In normal computation, face decomposition and frustum culling, each CUDA thread is responsible for the computation of a small set of faces, while in point projection and rendering area computation, each thread works on a different bunch of points.

All these stages are coded into different methods which use the following function in order to properly distribute the work among all the threads:

```c++
/**
 * This method returns the starting point and the number of elements on which the
 * calling thread should work with, in a parallel computation.
 * 
 * Parameters:
 * 	- work_size is the total number of elements
 * 	- skip corresponds to the size of each single element
 * 	- start will contain the starting point
 * 	- nloc will contain the number of elements to process
 */
inline __device__ void distribute(int work_size, int skip, int* start, int* nloc) {
    int threads = gridDim.x*blockDim.x;
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    int scaled_size = work_size/skip;
    int loc = scaled_size/threads;
    int carry = scaled_size%threads;
    if (index < carry) {
        *nloc = (loc+1)*skip;
        *start = *nloc*index;
    } else {
        *nloc = loc*skip;
        *start = *nloc*index + carry*skip;
    }
}
```

> The algorithm simply distributes the work among all the threads, by managing any combination of input size and threads number.

#### Rasterization
The rasterization process fits a texture on the faces of the object. It is done through the linear interpolation of the UV coordinates of the triangle vertices, according to the X and Y coordinates of the screen. On this stage, parallelization occurs on the pixels of the screen, so each thread works on a different set of pixels and for each one, it loops through the faces of the object and computes the color that the texture assumes on that interpolated point. The final value of the pixel will be the color taken from the face which is the closest to the camera on the $z$ axis.

## Performance comparisons
The engine has been tested with a AMD Ryzen 3600 CPU and a MSI Geforce GTX 1650 Super graphic card. FPS (Frames Per Second) is a good metric to evaluate the performances of the engine, it indeed counts the number of frames that the engine can process in one second. The code which computes this metric is executed on each frame and it is the following:
```c++
long now = nanoTime(); // Get current time in nanoseconds
long passedTime = now-lastUpdate; // Time passed since the previous frame
unprocessedTime += passedTime; // Update the number of unprocessed nanoseconds
lastUpdate = now; // Update the time of the last frame

fps++; // Increase the FPS counter
if (unprocessedTime >= 1E9) { // Check if unprocessed time exceeds one second
    showFPS(); // Show FPS value on the title bar
    unprocessedTime -= 1E9; // The last second has been processed
    fps = 0; // Reset FPS counter
}
```

The following simulation shows the performances of the engine before its optimization through CUDA.

![cpu-demo](images/cpu_demo.gif)
The *lag* is quite evident when the camera gets closer to the object in the scene (a rotating cube in this case), highlighted by the drastic drop in FPS (displayed in the window title bar), because both the number of pixels involved in the rendering and the load of the processor increase.

The following table shows the computational times (in milliseconds) for each pipeline stage averaged over the number of rendered frames:

| Stage | Averaged Computational Time (ms) |
|:-:|:-:|
| Points viewing | 0.002131 |
| Normal computation | 0.001511 |
| Faces decomposition | 0.001399 |
| Faces clipping | 0.001160 |
| Projection | 0.000127 |
| Render Area Computation | 0.002131 |
| Rasterization | 12.593671 |
| **Total frame processing** | **13.120979** |

As we can see from the table, most of the load is due to the rasterization process, while other computational times depend on input data sizes, which are very small for a cube, so they remain low.

![gpu-demo](images/gpu_demo.gif)
In the above simulation, we can see the gain in terms of FPS and movements smoothness thanks to the help of the GPU which performs the most of the work regarding the processing of the model and its rendering on the screen. When the camera gets closer to the object, FPS experience a slight drop but smoothness remains the same.

| Stage | Averaged Computational Time (ms) |
|:-:|:-:|
| Points viewing | 0.715701 |
| Normal computation | 0.182957 |
| Faces decomposition | 0.032343 |
| Faces clipping | 0.044581 |
| Projection | 0.005803 |
| Render Area Computation | 0.715701 |
| Rasterization | 1.411565 |
| **Total frame processing** | **3.720839** |

From the table above, we can see that the computational time needed for rasterization has dropped dramatically, but all the other stages have slowed down due to the overhead caused by device memory operations and kernel calls. However, the total computational time to process a single frame stays low, therefore we achieved a gain in terms of performances.

## Future updates
The engine can grow significantly in terms of provided features and performance optimizations. The first feature that can be added could be shaders, since normal computation is already implementd. To implement this feature, the engine needs some more informations about the materials of the meshes (e.i. how they absorbe, diffuse and reflect the lights). These informations are usually stored into *Material* files that have extension `.mtl`. So it would be necessary to add a function able of parsing those files and then implement some shading algorithms like *Phong* shading model.

In terms of optimizations, we can implement some algorithms to find the best trade off values for grid size and block size in each CUDA kernel, since the actual values are the results of some empirical tests. Moreover, the GPU memory could be managed better, for example using matrix allocation with pitch.