#ifndef _aspect_material_model_syla_h
#define _aspect_material_model_syla_h

#include <aspect/material_model/interface.h>
#include <aspect/material_model/equation_of_state/thermodynamic_table_lookup.h>
#include <aspect/simulator_access.h>
#include <deal.II/fe/component_mask.h>
namespace aspect
{
    namespace MaterialModel
    {
        using namespace dealii;

        /*My first material model trial. 
        Viscosity data is read from a profile.*/
        namespace internal
        {
            class ViscosityLookup
            {
                public:
                    ViscosityLookup(const std::string &filename,
                    const MPI_Comm &comm);

                    double _viscosity(double depth) const;
                private:
                    std::vector<double> values;

                    double min_depth;
                    double delta_depth;
                    double max_depth;
            };
        }
        template <int dim>
        class Syla : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
        {
            public:
                void
                initialize () override;

                //void update() override;

                virtual double viscosity (const double                  temperature,
                                          const double                  pressure,
                                          const std::vector<double>    &compositional_fields,
                                          const SymmetricTensor<2,dim> &strain_rate,
                                          const Point<dim>             &position) const;

                bool is_compressible () const override;

                void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                MaterialModel::MaterialModelOutputs<dim> &out) const override;

                static
                void
                declare_parameters (ParameterHandler &prm);

                void
                parse_parameters (ParameterHandler &prm) override;

                void
                create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const override;

            private:
                /**
                 * Whether the compositional fields representing mass fractions
                 * should be normalized to one when computing their fractions
                 * (if false), or whether there is an additional composition
                 * (the background field) that is not represented by a
                 * compositional field, and makes up the remaining fraction of
                 * material if the compositional fields add up to less than one
                 * at any given location (if true).
                 */
                bool has_background_field;

                /**
                 * Pointer to a composition mask, which is meant to be filled with
                 * one entry per compositional field that determines if this
                 * field is considered to represent a mass fractions (if the entry
                 * is set to true) or not (if set to false). This is needed for
                 * averaging of material properties.
                 */
                std::unique_ptr<ComponentMask> composition_mask;


                double thermal_conductivity_value;
                double min_eta;
                double max_eta;
                std::string data_directory;
                std::string _viscosity_file_name;
                std::unique_ptr<internal::ViscosityLookup> _viscosity_lookup;
                EquationOfState::ThermodynamicTableLookup<dim> equation_of_state;

                void fill_prescribed_outputs (const unsigned int i,
                                      const std::vector<double> &volume_fractions,
                                      const MaterialModel::MaterialModelInputs<dim> &in,
                                      MaterialModel::MaterialModelOutputs<dim> &out) const;
        };
    }
}

#endif