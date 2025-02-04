<!-- toc start: 3 [do not erase this comment] -->
**Table of contents**
- [CUDA Graphic Engine (CGE)](#cuda-graphic-engine-cge)
	- [Introduction](#introduction)
	- [Framework and libraries](#framework-and-libraries)
	- [Functioning](#functioning)
	- [Input data](#input-data)
		- [Model files](#model-files)
		- [Textures](#textures)
	- [Pipeline](#pipeline)
	- [CUDA optimizations](#cuda-optimizations)
		- [Points viewing](#points-viewing)
		- [Other pipeline stages](#other-pipeline-stages)
		- [Rastering](#rastering)
	- [Performance comparisons](#performance-comparisons)
	- [Build and run](#build-and-run)
<!-- toc end [do not erase this comment] -->

# CUDA Graphic Engine (CGE)

## Introduction
The project aims to create a graphic engine based on CUDA able of rendering 3D objects on the screen, using perspective projection to tranform 3D points into two dimensions. The application can also fit images (called **textures**) on the surfaces of the 3D objects, so it is possible to visualize any kind of stuff: objects, animals or even people, in a virtual world. The engine also gives to the user the opportunity to move around this world by interacting with the mouse and the keyboard.

## Framework and libraries
The engine is totally written in C++, using the CUDA framework to optimize the computations through parallelization on the GPU. The **cuBLAS** library is also used to simplify matrix operations, while the SDL library is responsible for managing the graphic and the inputs coming from the keyboard and the mouse.

## Functioning
In order to visualize a 3D object on the screen, the engine needs some informations about its structure and composition. Those data are stored in the **model** of the object, which provides three main informations:
- The relative positions of all the 3D points of the object, stored into a matrix of type `double`;
- The **UV coordinates** for texture rastering, stored into a matrix of type `double`. They consist of 2D vectors taking values between $0$ and $1$, which correspond to specific points of the texture, that can be whatever image of any dimension.
- **Faces** information: the 3D object consists of a bunch of triangles in the 3D space, described by their three vertices. So the model must specify how the points are connected to each other in order to construct the lattice of triangles that shapes the object. These data are stored into a matrix of `int` types corresponding to the indices of the points which make the triangles.

Another information needed if we want to fit a texture on the faces of the object is the matrix of indices of the texture points associated to each face point. Indeed, this matrix has the same dimensions of the face points matrix and it's of the same type.

On each frame, the engine takes each object in the world (technically called **mesh**) and performs some transformations on its points in order to **view** them, that is rotating and shifting the points in the 3D space in order to match the position and the rotation of the **camera**, giving the illusion of moving around the world.

After those transformations, the engine computes some other informations needed during the **rastering** of the texture.

## Input data
### Model files
Model information are usually stored into structured files that follow a specific standard and have the `.obj` extension. These files contain all the three (or four) informations that the engine needs in order to show the 3D object on the screen. The information are stored line by line, where the token starting each line indicates the kind of information written on that line:
- Lines starting with the **`v`** character contains the coordinates of a point, each separated by a whitespace.
- Lines starting with **`vt`** contains one texture point, described by its UV coordinates separated by a whitespace.
- Lines starting with the **`f`** character indicates a *face* of the object. It usually consists of a list of 3 or 4 positive integers separated by a whitespace, which stands for the indices of the points involved in that face. In order to associate each face point to a specific point of the texture (so that the texture can be correctly fitted on it), each index is flanked with the index of the related texture point, separated by a `/` character, for example: `f 1/1 2/2 3/3`.

### Textures
Texture are nothing else than square images. The more large are their dimensions, the more sharp will be the graphic of the engine, at the expense of the performances.

## Pipeline
The term ***pipeline*** refers to the whole process of transforming the points and perform computations up to the rendering of meshes on the screen. Let's see all of its stages more in detail:
1. **Points viewing**: on this stage, the points of the mesh currently processed are firstly rotated according to the rotation angles of the mesh. This allows the mesh to be rotated on itself, but only if the model refers to a cartesian system having its origin on the center point of the object. Then the world position of the object is added to each one of its points, while the camera position is subtracted from them. This is because of the illusion of moving around the world which involves the object to always move in the opposite direction of the camera. The same thing happens for the rotation of the object according to the opposite of the rotation angles of the camera.
1. **Normals computations**: normals are unit vectors which indicates the facing of each triangle of the mesh. They are useful to check which faces are in front of the camera and should be rendered, and which ones are hidden instead. This implies an optimization step since invisible faces can be discarded in order to avoid to render them and save computational time. Normals, can also be used to apply shading which refers to how the lights interact with the material of the mesh.
To compute the normals, the engine performs the cross product between two sides of each triangle, then it sums the computed vector to the normals corresponding to each point of the triangle. This way, if we wanted to add shading, the transition between one face and the other would be much smoother.
1. **Face decomposition**: on this stage, the processed points are inserted into a list of vertices which follows the connections between the points in order to construct the faces as specified by the model. This way, we assure that to each three vertices corresponds a face of the object. Face decomposition also regards other data of the mesh, so to each vertex is associated the corresponding normal and the texture point. During this stage, we also check for visible faces and discard the invisible ones.
1. **Face clipping**: clipping of the faces is needed in order to limit the rendering to a small portion of the world, which corresponds to that captured by the camera. This avoids an overload of the engine when the world gets too big. To do that the engine generate a **frustrum**, that is a set of planes in the 3D space, described by their position and normal, which delimit the field of view. Each triangle which crosses a plane, is clipped against that plane into one or two smaller triangles.
1. The last data needed by the renderer are the projections of each 3D point in the 2D space. So for each vertex, the engine computes its corresponding projection by following this formulas:
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

For performance improvement purposes, the pipeline also performs another computation, that is the *rendering area*. Since each mesh occupies a small portion of the screen on the average, is clever to find the minimum rectangle which contains all the faces of the mesh. This can be done by looping through all the projected points of the mesh and search for the top left and the bottom right corners of the rectangle.

## CUDA optimizations
In order to reduce the load of work that the CPU should face, the engine uses the CUDA framework as the interface for the GPU so that it can use parallelization to accelerate the computation of the heaviest jobs. Let's see where and how CUDA is used.

### Points viewing
In the stage of points viewing, points must be rotated and translated. For the rotation part, we have to multiply the point matrix with the rotation matrix, so we can use the cuBLAS library and the `cublasDgemm` method to perform this computation. The translation part, simply consists of a summation of the shift vector to the rows of the point matrix, so we can build a kernel which parallelizes this computation on the rows of the matrix. The same thing can be done for the subtraction of the camera position vector from the rows of the point matrix.

### Other pipeline stages
All other stages of the pipeline use CUDA to accelerate the computation through kernels which parallelize the work on the first dimension of the considered matrix. In normal computation, face decomposition and clipping, each CUDA thread is responsible for the computation of a small set of faces, while in point projection and rendering area computation, each thread works on a different bunch of points.

### Rastering
The rastering process fits a texture on the faces of the object. It is done through the linear interpolation of the UV coordinates of the triangle vertices, according to the X and Y coordinates of the screen. On this stage, the parallelization is done on the pixels of the screen, so each thread works on a different set of pixels and for each one loops through the faces of the object and computes the color that the texture assumes on that interpolated point. The final value of the pixel will be the color taken from the face which is the closest to the camera on the $z$ axis.

## Performance comparisons

## Build and run
```
nvcc sources/*.cu -o cge -lcublas -lSDL2 -lSDL2_image -arch=sm_xx
./cge
```
> The engine only works on GPUs with SM 6.0 or higher.