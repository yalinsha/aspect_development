#include <aspect/material_model/syla.h>
#include <aspect/material_model/equation_of_state/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/table.h>
#include <fstream>
#include <iostream>
#include <memory>

namespace aspect
{
    namespace MaterialModel
    {
        namespace internal
        {
            ViscosityLookup::ViscosityLookup(const std::string &filename,
            const MPI_Comm &comm)
            {
                std::string temp;
                // Read data from disk and distribute among processes
                std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

                min_depth=1e20;
                max_depth=-1;

                while (!in.eof())
                {
                    double visc, depth;
                    in >> visc;
                    if (in.eof())
                    break;
                    in >> depth;
                    depth *=1000.0;
                    std::getline(in, temp);

                    min_depth = std::min(depth, min_depth);
                    max_depth = std::max(depth, max_depth);

                    values.push_back(visc);
                }
                delta_depth = (max_depth-min_depth)/(values.size()-1);
            }

            double ViscosityLookup::_viscosity(double depth) const
            {
                depth=std::max(min_depth, depth);
                depth=std::min(depth, max_depth);

                Assert(depth>=min_depth, ExcMessage("ASPECT found a depth less than min_depth."));
                Assert(depth<=max_depth, ExcMessage("ASPECT found a depth greater than max_depth."));
                const unsigned int idx = static_cast<unsigned int>((depth-min_depth)/delta_depth);
                Assert(idx<values.size(), ExcMessage("Attempting to look up a depth with an index that would be out of range. (depth-min_depth)/delta_depth too large."));
                return values[idx];
            }
        }

        template <int dim>
        void
        Syla<dim>::initialize()
        {
            equation_of_state.initialize();

            _viscosity_lookup
                = std::make_unique<internal::ViscosityLookup>(data_directory+_viscosity_file_name,
                                                                    this->get_mpi_communicator());
        }

        template <int dim>
        double
        Syla<dim>::
        viscosity (const double temperature,
                const double /*pressure*/,
                const std::vector<double> &,
                const SymmetricTensor<2,dim> &,
                const Point<dim> &position) const
        {
            const double depth = this->get_geometry_model().depth(position);
            //const double adiabatic_temperature = this->get_adiabatic_conditions().temperature(position);

            const double vis_ = _viscosity_lookup->_viscosity(depth);
            const double E_d_R = 160000/8.3145;
            const double T_ref = 1573.0;
            const double vis_out = vis_*std::exp(E_d_R*(1.0/temperature-1.0/T_ref));

            return std::max(std::min(vis_out,max_eta),min_eta);
        }

        template <int dim>
        bool
        Syla<dim>::
        is_compressible () const
        {
            return equation_of_state.is_compressible ();
        }

        template <int dim>
        void
        Syla<dim>::
        evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                MaterialModel::MaterialModelOutputs<dim> &out) const
        {
            std::vector<EquationOfStateOutputs<dim>> eos_outputs (in.n_evaluation_points(), equation_of_state.number_of_lookups());
            std::vector<std::vector<double>> volume_fractions (in.n_evaluation_points(), std::vector<double> (equation_of_state.number_of_lookups()));

            // We need to make a copy of the material model inputs because we want to use the adiabatic pressure
            // rather than the real pressure for the equations of state (to avoid numerical instabilities).
            MaterialModel::MaterialModelInputs<dim> eos_in(in);
            for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
                eos_in.pressure[i] = this->get_adiabatic_conditions().pressure(in.position[i]);

            // Evaluate the equation of state properties over all evaluation points
            equation_of_state.evaluate(eos_in, eos_outputs);

            for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
                {
                if (in.requests_property(MaterialProperties::viscosity))
                    out.viscosities[i] = viscosity(in.temperature[i], in.pressure[i], in.composition[i], in.strain_rate[i], in.position[i]);

                out.thermal_conductivities[i] = thermal_conductivity_value;
                for (unsigned int c=0; c<in.composition[i].size(); ++c)
                    out.reaction_terms[i][c] = 0;

                // Calculate volume fractions from mass fractions
                // If there is only one lookup table, set the mass and volume fractions to 1
                std::vector<double> mass_fractions;
                if (equation_of_state.number_of_lookups() == 1)
                    mass_fractions.push_back(1.0);
                else
                    {
                    // We only want to compute mass/volume fractions for fields that are chemical compositions.
                    std::vector<double> chemical_compositions;
                    const std::vector<typename Parameters<dim>::CompositionalFieldDescription> composition_descriptions = this->introspection().get_composition_descriptions();

                    for (unsigned int c=0; c<in.composition[i].size(); ++c)
                        if (composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::chemical_composition
                            || composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::unspecified)
                        chemical_compositions.push_back(in.composition[i][c]);

                    mass_fractions = MaterialUtilities::compute_composition_fractions(chemical_compositions, *composition_mask);

                    // The function compute_volumes_from_masses expects as many mass_fractions as densities.
                    // But the function compute_composition_fractions always adds another element at the start
                    // of the vector that represents the background field. If there is no lookup table for
                    // the background field, the mass_fractions vector is too long and we remove this element.
                    if (!has_background_field)
                        mass_fractions.erase(mass_fractions.begin());
                    }

                volume_fractions[i] = MaterialUtilities::compute_volumes_from_masses(mass_fractions,
                                                                                    eos_outputs[i].densities,
                                                                                    true);

                MaterialUtilities::fill_averaged_equation_of_state_outputs(eos_outputs[i], mass_fractions, volume_fractions[i], i, out);
                fill_prescribed_outputs(i, volume_fractions[i], in, out);
                
                /*out.densities[i] = eos_outputs[i].densities[0];
                out.thermal_expansion_coefficients[i] = eos_outputs[i].thermal_expansion_coefficients[0];
                out.specific_heat[i] = eos_outputs[i].specific_heat_capacities[0];
                out.compressibilities[i] = eos_outputs[i].compressibilities[0];*/
                }

            // fill additional outputs if they exist
            equation_of_state.fill_additional_outputs(in, volume_fractions, out);
            /*
            // The trial model does not depend on composition
            EquationOfStateOutputs<dim> eos_outputs (1);

            for (unsigned int i=0; i<in.n_evaluation_points(); ++i)
            {
                equation_of_state.evaluate(in, i, eos_outputs);

                out.viscosities[i] = viscosity(in.temperature[i], in.pressure[i], in.composition[i], in.strain_rate[i], in.position[i]);
                //out.viscosities[i] = constant_rheology.compute_viscosity();
                out.densities[i] = eos_outputs.densities[0];
                out.thermal_expansion_coefficients[i] = eos_outputs.thermal_expansion_coefficients[0];
                out.specific_heat[i] = eos_outputs.specific_heat_capacities[0];
                out.thermal_conductivities[i] = thermal_conductivity_value;
                out.compressibilities[i] = eos_outputs.compressibilities[0];

                for (unsigned int c=0; c<in.composition[i].size(); ++c)
                    out.reaction_terms[i][c] = 0.0;
            }*/
        }

        template <int dim>
        void
        Syla<dim>::
        fill_prescribed_outputs(const unsigned int q,
                                const std::vector<double> &,
                                const MaterialModel::MaterialModelInputs<dim> &,
                                MaterialModel::MaterialModelOutputs<dim> &out) const
        {
        // set up variable to interpolate prescribed field outputs onto compositional field
        PrescribedFieldOutputs<dim> *prescribed_field_out = out.template get_additional_output<PrescribedFieldOutputs<dim>>();

        if (this->introspection().composition_type_exists(Parameters<dim>::CompositionalFieldDescription::density)
            && prescribed_field_out != nullptr)
            {
            const unsigned int projected_density_index = this->introspection().find_composition_type(Parameters<dim>::CompositionalFieldDescription::density);
            prescribed_field_out->prescribed_field_outputs[q][projected_density_index] = out.densities[q];
            }
        }

        template <int dim>
        void
        Syla<dim>::declare_parameters (ParameterHandler &prm)
        {
        prm.enter_subsection("Material model");
        {
            prm.enter_subsection("Syla model");
            {
            prm.declare_entry ("Data directory", "$ASPECT_SOURCE_DIR/data/material-model/syla/",
                                Patterns::DirectoryName (),
                                "The path to the model data. The path may also include the special "
                                "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                                "in which the ASPECT source files were located when ASPECT was "
                                "compiled. This interpretation allows, for example, to reference "
                                "files located in the `data/' subdirectory of ASPECT. ");
            prm.declare_entry ("Viscosity file name", "syl-visc.txt",
                                Patterns::Anything (),
                                "The file name of the viscosity data. ");
            prm.declare_entry ("Thermal conductivity", "4.7",
                                Patterns::Double (0.),
                                "The value of the thermal conductivity $k$. "
                                "Units: \\si{\\watt\\per\\meter\\per\\kelvin}.");
            prm.declare_entry ("Minimum viscosity", "1e18",
                                Patterns::Double (0.),
                                "The minimum viscosity that is allowed in the viscosity "
                                "calculation. Smaller values will be cut off.");
            prm.declare_entry ("Maximum viscosity", "5e24",
                                Patterns::Double (0.),
                                "The maximum viscosity that is allowed in the viscosity "
                                "calculation. Larger values will be cut off.");
            //Rheology::ConstantViscosity::declare_parameters(prm,5e24);
            EquationOfState::ThermodynamicTableLookup<dim>::declare_parameters(prm);
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        }
        template <int dim>
        void
        Syla<dim>::parse_parameters (ParameterHandler &prm)
        {
            prm.enter_subsection("Material model");
            {
                prm.enter_subsection("Syla model");
                {
                    data_directory = Utilities::expand_ASPECT_SOURCE_DIR(prm.get ("Data directory"));
                    _viscosity_file_name   = prm.get ("Viscosity file name");
                    thermal_conductivity_value = prm.get_double ("Thermal conductivity");
                    min_eta              = prm.get_double ("Minimum viscosity");
                    max_eta              = prm.get_double ("Maximum viscosity");
                    //equation_of_state.initialize_simulator (this->get_simulator());

                    // Parse the table lookup parameters
                    equation_of_state.initialize_simulator (this->get_simulator());
                    equation_of_state.parse_parameters(prm);

                    // Check if compositional fields represent a composition
                    const std::vector<typename Parameters<dim>::CompositionalFieldDescription> composition_descriptions = this->introspection().get_composition_descriptions();

                    // All chemical compositional fields are assumed to represent mass fractions.
                    // If the field type is unspecified (has not been set in the input file),
                    // we have to assume it also represents a chemical composition for reasons of
                    // backwards compatibility.
                    composition_mask = std::make_unique<ComponentMask> (this->n_compositional_fields(), false);
                    for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                        if (composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::chemical_composition
                            || composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::unspecified)
                        composition_mask->set(c, true);

                    const unsigned int n_chemical_fields = composition_mask->n_selected_components();

                    // Assign background field and do some error checking
                    AssertThrow ((equation_of_state.number_of_lookups() == 1) /*||
                                (equation_of_state.number_of_lookups() == n_chemical_fields) ||
                                (equation_of_state.number_of_lookups() == n_chemical_fields + 1)*/,
                                ExcMessage(/*"The Steinberger material model assumes that all compositional "
                                            "fields of the type chemical composition correspond to mass fractions of "
                                            "materials. There must either be one material lookup file, the same "
                                            "number of material lookup files as compositional fields of type chemical "
                                            "composition, or one additional file (if a background field is used). You "
                                            "have "
                                            + Utilities::int_to_string(equation_of_state.number_of_lookups())
                                            + " material data files, but there are "
                                            + Utilities::int_to_string(n_chemical_fields)
                                            + " fields of type chemical composition."*/
                                            "We only deal with one compositional field for now."));

                    has_background_field = (equation_of_state.number_of_lookups() == n_chemical_fields + 1);
                }
                prm.leave_subsection();
            }
            prm.leave_subsection();

            this->model_dependence.viscosity = NonlinearDependence::temperature;
            this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
            this->model_dependence.compressibility = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
            this->model_dependence.specific_heat = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
            this->model_dependence.thermal_conductivity = NonlinearDependence::none;
        }
        template <int dim>
        void
        Syla<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
        {
        equation_of_state.create_additional_named_outputs(out);

        if (this->introspection().composition_type_exists(Parameters<dim>::CompositionalFieldDescription::density)
            && out.template get_additional_output<PrescribedFieldOutputs<dim>>() == nullptr)
            {
            const unsigned int n_points = out.n_evaluation_points();
            out.additional_outputs.push_back(
                std::make_unique<MaterialModel::PrescribedFieldOutputs<dim>> (n_points, this->n_compositional_fields()));
            }
        }
    }
}


namespace aspect
{
    namespace MaterialModel
    {
        ASPECT_REGISTER_MATERIAL_MODEL(Syla,
        "syla",
        "My first trial model")
    }
}
