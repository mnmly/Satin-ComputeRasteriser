# Third-Party Notices

This project is a Swift/Satin/Metal port of the batch-oriented compute rasterization approach from Markus Schuetz's `compute_rasterizer` project.

## compute_rasterizer

- Upstream: `https://github.com/m-schuetz/compute_rasterizer`
- Local source used during this port: `/Users/mnmly/Development-local/GitHub/cpp/compute_rasterizer`
- Upstream commit used as reference: `f2cbb65` on `master`
- Primary referenced modules:
  - `modules/compute_loop_las_hqs`
  - `modules/compute/LasLoaderSparse.*`

Upstream license:

```text
Copyright 2022 Markus Schuetz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The upstream project notes that some shader files adapt frustum-culling code from three.js under the MIT license. This port keeps the same algorithmic lineage in its Metal frustum culling helpers.

## Satin

- Upstream: `https://github.com/mnmly/Satin`
- Package dependency: branch `feature/2.0-shader-source-transforms`
- License: MIT
- Copyright: `Copyright (c) 2025 Hi-Rez`

Satin is used as the rendering, Metal view, scene graph, and compute processor framework.

## Research References

If you use this work in research or published material, cite the original compute rasterization papers:

```bibtex
@article{SCHUETZ-2022-PCC,
  title = {Software Rasterization of 2 Billion Points in Real Time},
  author = {Markus Schuetz and Bernhard Kerbl and Michael Wimmer},
  year = {2022},
  month = jul,
  journal = {Proc. ACM Comput. Graph. Interact. Tech.},
  volume = {5},
  pages = {1--16},
  URL = {https://www.cg.tuwien.ac.at/research/publications/2022/SCHUETZ-2022-PCC/}
}

@article{SCHUETZ-2021-PCC,
  title = {Rendering Point Clouds with Compute Shaders and Vertex Order Optimization},
  author = {Markus Schuetz and Bernhard Kerbl and Michael Wimmer},
  year = {2021},
  month = jul,
  doi = {10.1111/cgf.14345},
  journal = {Computer Graphics Forum},
  number = {4},
  volume = {40},
  pages = {115--126},
  keywords = {point-based rendering, compute shader, real-time rendering},
  URL = {https://www.cg.tuwien.ac.at/research/publications/2021/SCHUETZ-2021-PCC/}
}
```

