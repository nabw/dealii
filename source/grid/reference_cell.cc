// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/polynomial.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/polynomials.h>
#include <deal.II/simplex/quadrature_lib.h>

DEAL_II_NAMESPACE_OPEN


namespace ReferenceCell
{
  namespace internal
  {
    dealii::ReferenceCell::Type
    make_reference_cell_from_int(const std::uint8_t kind)
    {
      // Make sure these are the only indices from which objects can be
      // created.
      Assert((kind == static_cast<std::uint8_t>(-1)) || (kind < 8),
             ExcInternalError());

      // Call the private constructor, which we can from here because this
      // function is a 'friend'.
      return {kind};
    }
  } // namespace internal


  const Type Type::Vertex  = internal::make_reference_cell_from_int(0);
  const Type Type::Line    = internal::make_reference_cell_from_int(1);
  const Type Type::Tri     = internal::make_reference_cell_from_int(2);
  const Type Type::Quad    = internal::make_reference_cell_from_int(3);
  const Type Type::Tet     = internal::make_reference_cell_from_int(4);
  const Type Type::Pyramid = internal::make_reference_cell_from_int(5);
  const Type Type::Wedge   = internal::make_reference_cell_from_int(6);
  const Type Type::Hex     = internal::make_reference_cell_from_int(7);
  const Type Type::Invalid =
    internal::make_reference_cell_from_int(static_cast<std::uint8_t>(-1));



  std::string
  Type::to_string() const
  {
    if (*this == Vertex)
      return "Vertex";
    else if (*this == Line)
      return "Line";
    else if (*this == Tri)
      return "Tri";
    else if (*this == Quad)
      return "Quad";
    else if (*this == Tet)
      return "Tet";
    else if (*this == Pyramid)
      return "Pyramid";
    else if (*this == Wedge)
      return "Wedge";
    else if (*this == Hex)
      return "Hex";
    else if (*this == Invalid)
      return "Invalid";

    Assert(false, ExcNotImplemented());

    return "Invalid";
  }



  template <int dim, int spacedim>
  std::unique_ptr<Mapping<dim, spacedim>>
  Type::get_default_mapping(const unsigned int degree) const
  {
    AssertDimension(dim, get_dimension());

    if (is_hyper_cube())
      return std::make_unique<MappingQGeneric<dim, spacedim>>(degree);
    else if (is_simplex())
      return std::make_unique<MappingFE<dim, spacedim>>(
        Simplex::FE_P<dim, spacedim>(degree));
    else if (*this == Type::Pyramid)
      return std::make_unique<MappingFE<dim, spacedim>>(
        Simplex::FE_PyramidP<dim, spacedim>(degree));
    else if (*this == Type::Wedge)
      return std::make_unique<MappingFE<dim, spacedim>>(
        Simplex::FE_WedgeP<dim, spacedim>(degree));
    else
      {
        Assert(false, ExcNotImplemented());
      }

    return std::make_unique<MappingQGeneric<dim, spacedim>>(degree);
  }



  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  Type::get_default_linear_mapping() const
  {
    AssertDimension(dim, get_dimension());

    if (is_hyper_cube())
      {
        return StaticMappingQ1<dim, spacedim>::mapping;
      }
    else if (is_simplex())
      {
        static const MappingFE<dim, spacedim> mapping(
          Simplex::FE_P<dim, spacedim>(1));
        return mapping;
      }
    else if (*this == Type::Pyramid)
      {
        static const MappingFE<dim, spacedim> mapping(
          Simplex::FE_PyramidP<dim, spacedim>(1));
        return mapping;
      }
    else if (*this == Type::Wedge)
      {
        static const MappingFE<dim, spacedim> mapping(
          Simplex::FE_WedgeP<dim, spacedim>(1));
        return mapping;
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    return StaticMappingQ1<dim, spacedim>::mapping; // never reached
  }



  template <int dim, int spacedim>
  const Mapping<dim, spacedim> &
  get_default_linear_mapping(const Triangulation<dim, spacedim> &triangulation)
  {
    const auto &reference_cell_types = triangulation.get_reference_cell_types();
    Assert(reference_cell_types.size() == 1,
           ExcMessage(
             "This function can only work for triangulations that "
             "use only a single cell type -- for example, only triangles "
             "or only quadrilaterals. For mixed meshes, there is no "
             "single linear mapping object that can be used for all "
             "cells of the triangulation. The triangulation you are "
             "passing to this function uses multiple cell types."));

    return reference_cell_types.front()
      .template get_default_linear_mapping<dim, spacedim>();
  }



  template <int dim>
  Quadrature<dim>
  Type::get_gauss_type_quadrature(const unsigned n_points_1D) const
  {
    AssertDimension(dim, get_dimension());

    if (is_hyper_cube())
      return QGauss<dim>(n_points_1D);
    else if (is_simplex())
      return Simplex::QGauss<dim>(n_points_1D);
    else if (*this == Type::Pyramid)
      return Simplex::QGaussPyramid<dim>(n_points_1D);
    else if (*this == Type::Wedge)
      return Simplex::QGaussWedge<dim>(n_points_1D);
    else
      Assert(false, ExcNotImplemented());

    return Quadrature<dim>(); // never reached
  }



  template <int dim>
  const Quadrature<dim> &
  Type::get_nodal_type_quadrature() const
  {
    AssertDimension(dim, get_dimension());

    // A function that is used to fill a quadrature object of the
    // desired type the first time we encounter a particular
    // reference cell
    const auto create_quadrature = [](const Type &reference_cell) {
      Triangulation<dim> tria;
      GridGenerator::reference_cell(reference_cell, tria);

      return Quadrature<dim>(tria.get_vertices());
    };

    if (is_hyper_cube())
      {
        static const Quadrature<dim> quadrature = create_quadrature(*this);
        return quadrature;
      }
    else if (is_simplex())
      {
        static const Quadrature<dim> quadrature = create_quadrature(*this);
        return quadrature;
      }
    else if (*this == Type::Pyramid)
      {
        static const Quadrature<dim> quadrature = create_quadrature(*this);
        return quadrature;
      }
    else if (*this == Type::Wedge)
      {
        static const Quadrature<dim> quadrature = create_quadrature(*this);
        return quadrature;
      }
    else
      Assert(false, ExcNotImplemented());

    static const Quadrature<dim> dummy;
    return dummy; // never reached
  }

#include "reference_cell.inst"

} // namespace ReferenceCell

DEAL_II_NAMESPACE_CLOSE
