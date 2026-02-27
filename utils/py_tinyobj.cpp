// This file is a Python wrapper for the TinyOBJLoader C++ library using pybind11.
#define TINYOBJLOADER_IMPLEMENTATION
#include "TinyObjLoader.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cassert>

namespace py = pybind11;

PYBIND11_MODULE(tinyobjloader_py, m)
{
    m.def("load_obj",
          [](const std::string &filename, const std::string &mtl_dir)
          {
              // initialize loader variables
              tinyobj::attrib_t attrib;
              std::vector<tinyobj::shape_t> shapes;
              std::vector<tinyobj::material_t> materials;
              std::string warn, err;

              // load obj with triangulation enabled
              bool ret = tinyobj::LoadObj(
                  &attrib, &shapes, &materials, &warn, &err,
                  filename.c_str(), mtl_dir.c_str(), true);

              // check for errors
              if (!ret)
                  throw std::runtime_error(err);
              assert(attrib.vertices.size() > 0);

              std::vector<int> indices;
              std::vector<int> material_ids;

              // loop over shapes
              for (const auto &shape : shapes)
              {
                  size_t index_offset = 0;

                  // loop over faces
                  for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
                  {
                      int fv = shape.mesh.num_face_vertices[f];
                      // !! I am expecting triangles sice triangulation enabled !!
                      assert(fv == 3);

                      // loop over vertices in the face
                      for (size_t v = 0; v < fv; v++)
                      {
                          tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                          indices.push_back(idx.vertex_index);
                      }

                      // get material index for this face
                      int mid = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
                      material_ids.push_back(mid);

                      index_offset += fv;
                  }
              }

              // pack materials into flat array
              size_t m_size = std::max<size_t>(1, materials.size());
              std::vector<float> mat_array(m_size * 10, 0.0f);

              for (size_t i = 0; i < materials.size(); i++)
              {
                  const auto &m = materials[i];
                  float *dst = &mat_array[i * 10];

                  // copy material properties
                  dst[0] = m.diffuse[0];
                  dst[1] = m.diffuse[1];
                  dst[2] = m.diffuse[2];
                  dst[3] = m.specular[0];
                  dst[4] = m.specular[1];
                  dst[5] = m.specular[2];
                  dst[6] = m.shininess;
                  dst[7] = m.emission[0];
                  dst[8] = m.emission[1];
                  dst[9] = m.emission[2];
              }

              // return tuple of raw data
              return py::make_tuple(attrib.vertices, indices, material_ids, mat_array);
          });
}
