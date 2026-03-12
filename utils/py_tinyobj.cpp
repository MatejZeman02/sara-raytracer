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

              if (!ret)
                  throw std::runtime_error(err);

              assert(attrib.vertices.size() > 0);

              std::vector<int> v_indices;
              std::vector<int> n_indices;
              std::vector<int> material_ids;

              // loop over shapes
              for (const auto &shape : shapes)
              {
                  size_t index_offset = 0;

                  // loop over faces
                  for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
                  {
                      int fv = shape.mesh.num_face_vertices[f];

                      assert(fv == 3);

                      // loop over vertices in the face
                      for (size_t v = 0; v < fv; v++)
                      {
                          tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                          v_indices.push_back(idx.vertex_index);
                          n_indices.push_back(idx.normal_index);
                      }

                      // get material index for this face
                      int mid = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
                      material_ids.push_back(mid);

                      index_offset += fv;
                  }
              }

              // pack materials into flat array of 14 floats
              size_t m_size = std::max<size_t>(1, materials.size());
              std::vector<float> mat_array(m_size * 14, 0.0f);

              for (size_t i = 0; i < materials.size(); i++)
              {
                  const auto &m = materials[i];
                  float *dst = &mat_array[i * 14];

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
                  dst[10] = m.transmittance[0];
                  dst[11] = m.transmittance[1];
                  dst[12] = m.transmittance[2];
                  dst[13] = m.ior;
              }

              // return tuple of raw data including normals
              return py::make_tuple(attrib.vertices, attrib.normals, v_indices, n_indices, material_ids, mat_array);
          });
}