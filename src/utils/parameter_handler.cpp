/**
From Emir Demirovic "MurTree"
https://bitbucket.org/EmirD/murtree
*/

#include "utils/parameter_handler.h"

namespace STreeD {
    std::string ParameterHandler::GetStringParameter(const std::string& parameter_name) const {
        auto iter = parameters_string_.find(parameter_name);
        if (iter == parameters_string_.end()) { std::cout << "Unknown string parameter: " << parameter_name << "\n"; exit(1); }
        return iter->second.current_value;
    }

    int64_t ParameterHandler::GetIntegerParameter(const std::string& parameter_name) const {
        auto iter = parameters_integer_.find(parameter_name);
        if (iter == parameters_integer_.end()) { std::cout << "Unknown integer parameter: " << parameter_name << "\n"; exit(1); }
        return iter->second.current_value;
    }

    bool ParameterHandler::GetBooleanParameter(const std::string& parameter_name) const {
        auto iter = parameters_boolean_.find(parameter_name);
        if (iter == parameters_boolean_.end()) { std::cout << "Unknown Boolean parameter: " << parameter_name << "\n"; exit(1); }
        return iter->second.current_value;
    }

    double ParameterHandler::GetFloatParameter(const std::string& parameter_name) const {
        auto iter = parameters_float_.find(parameter_name);
        if (iter == parameters_float_.end()) { std::cout << "Unknown float parameter: " << parameter_name << "\n"; exit(1); }
        return iter->second.current_value;
    }

    void ParameterHandler::DefineNewCategory(const std::string& category_name, const std::string short_description) {
        if (category_name.size() == 0) { std::cout << "Empty strings are not allowed for category names!\n"; exit(1); }
        if (std::find(categories_.begin(), categories_.end(), category_name) != categories_.end()) { std::cout << "Category with name " << category_name << " already exists!\n";  exit(1); }

        Category category;
        category.name = category_name;
        category.short_description = short_description;
        categories_.push_back(category);
    }

    void ParameterHandler::DefineStringParameter(const std::string& parameter_name, const std::string& short_description, const std::string& default_value,
                                            const std::string& category_name, const std::vector<std::string>& allowed_values, bool optional) {
        auto category_iterator = find(categories_.begin(), categories_.end(), category_name);
        if (category_iterator == categories_.end()) { std::cout << "Category " << category_name << " does not exist, it needs to be defined before the " << parameter_name << " parameter can be assinged to it!\n"; exit(1); }
        if (parameter_name.size() == 0) { std::cout << "Empty strings are not allowed for parameter names!\n"; exit(1); }
        //test if we already declared the parameter - a parameter should not be declared twice
        if (parameters_string_.count(parameter_name) == 1) { std::cout << "String parameter " << parameter_name << " already declared\n"; exit(1); }

        StringEntry entry;
        entry.name = parameter_name;
        entry.short_description = short_description;
        entry.category_name = category_name;
        entry.default_value = default_value;
        entry.current_value = default_value;
        entry.allowed_values = allowed_values;
        entry.optional = optional;
        parameters_string_[parameter_name] = entry;

        PairNameType category_entry;
        category_entry.name = parameter_name;
        category_entry.type = "string";
        category_iterator->parameters.push_back(category_entry);
    }

    void ParameterHandler::DefineIntegerParameter(const std::string& parameter_name, const std::string& short_description, int64_t default_value, const std::string& category_name, int64_t min_value, int64_t max_value) {
        auto category_iterator = find(categories_.begin(), categories_.end(), category_name);
        if (category_iterator == categories_.end()) { std::cout << "Category " << category_name << " does not exist, it needs to be defined before the " << parameter_name << " parameter can be assinged to it!\n"; exit(1); }
        if (parameter_name.size() == 0) { std::cout << "Empty strings are not allowed for parameter names!\n"; exit(1); }
        //test if we already declared the parameter - a parameter should not be declared twice
        if (parameters_integer_.count(parameter_name) == 1) { std::cout << "Integer parameter " << parameter_name << " already declared\n"; exit(1); }

        IntegerEntry entry;
        entry.name = parameter_name;
        entry.short_description = short_description;
        entry.category_name = category_name;
        entry.default_value = default_value;
        entry.current_value = default_value;
        entry.min_value = min_value;
        entry.max_value = max_value;
        parameters_integer_[parameter_name] = entry;

        PairNameType category_entry;
        category_entry.name = parameter_name;
        category_entry.type = "integer";
        category_iterator->parameters.push_back(category_entry);
    }

    void ParameterHandler::DefineBooleanParameter(const std::string& parameter_name, const std::string& short_description, bool default_value, const std::string& category_name) {
        auto category_iterator = find(categories_.begin(), categories_.end(), category_name);
        if (category_iterator == categories_.end()) { std::cout << "Category " << category_name << " does not exist, it needs to be defined before the " << parameter_name << " parameter can be assinged to it!\n"; exit(1); }
        if (parameter_name.size() == 0) { std::cout << "Empty strings are not allowed for parameter names!\n"; exit(1); }
        //test if we already declared the parameter - a parameter should not be declared twice
        if (parameters_integer_.count(parameter_name) == 1) { std::cout << "Boolean parameter " << parameter_name << " already declared\n"; exit(1); }

        BooleanEntry entry;
        entry.name = parameter_name;
        entry.short_description = short_description;
        entry.category_name = category_name;
        entry.default_value = default_value;
        entry.current_value = default_value;
        parameters_boolean_[parameter_name] = entry;

        PairNameType category_entry;
        category_entry.name = parameter_name;
        category_entry.type = "Boolean";
        category_iterator->parameters.push_back(category_entry);
    }

    void ParameterHandler::DefineFloatParameter(const std::string& parameter_name, const std::string& short_description, double default_value, const std::string& category_name, double min_value, double max_value) {
        auto category_iterator = find(categories_.begin(), categories_.end(), category_name);
        if (category_iterator == categories_.end()) { std::cout << "Category " << category_name << " does not exist, it needs to be defined before the " << parameter_name << " parameter can be assinged to it!\n"; exit(1); }
        if (parameter_name.size() == 0) { std::cout << "Empty strings are not allowed for parameter names!\n"; exit(1); }
        //test if we already declared the parameter - a parameter should not be declared twice
        if (parameters_float_.count(parameter_name) == 1) { std::cout << "Float parameter " << parameter_name << " already declared\n"; exit(1); }

        FloatEntry entry;
        entry.name = parameter_name;
        entry.short_description = short_description;
        entry.category_name = category_name;
        entry.default_value = default_value;
        entry.current_value = default_value;
        entry.min_value = min_value;
        entry.max_value = max_value;
        parameters_float_[parameter_name] = entry;

        PairNameType category_entry;
        category_entry.name = parameter_name;
        category_entry.type = "float";
        category_iterator->parameters.push_back(category_entry);
    }

    void ParameterHandler::SetStringParameter(const std::string& parameter_name, const std::string& new_value) {
        CheckStringParameter(parameter_name, new_value);
        parameters_string_[parameter_name].current_value = new_value;
    }

    void ParameterHandler::SetIntegerParameter(const std::string& parameter_name, int64_t new_value) {
        CheckIntegerParameter(parameter_name, new_value);
        parameters_integer_[parameter_name].current_value = new_value;
    }

    void ParameterHandler::SetBooleanParameter(const std::string& parameter_name, bool new_value) {
        CheckBooleanParameter(parameter_name, new_value);
        parameters_boolean_[parameter_name].current_value = new_value;
    }

    void ParameterHandler::SetFloatParameter(const std::string& parameter_name, double new_value) {
        CheckFloatParameter(parameter_name, new_value);
        parameters_float_[parameter_name].current_value = new_value;
    }

    void ParameterHandler::ParseCommandLineArguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; i += 2) {
            if (argv[i][0] != '-') { std::cout << "Parsing error: expected a parameter name with a '-' in the beggining, instead found this: " << argv[i][0] << "\n"; exit(1); }
            std::string name(argv[i] + 1); //skip the zero position character since it is expected to be a '-' character

            if (parameters_string_.count(name) == 1) {
                std::string value = argv[i + 1];
                SetStringParameter(name, value);
            } else if (parameters_integer_.count(name) == 1) {
                //todo check if the string is actually an integer
                int64_t value;
                std::stringstream(argv[i + 1]) >> value;
                SetIntegerParameter(name, value);
            } else if (parameters_boolean_.count(name) == 1) {
                //todo check if the string is actually a Boolean
                bool value;
                std::stringstream(argv[i + 1]) >> value;
                SetBooleanParameter(name, value);
            } else if (parameters_float_.count(name) == 1) {
                //todo check if the string is actually a floating point number
                double value;
                std::stringstream(argv[i + 1]) >> value;
                SetFloatParameter(name, value);
            } else {
                std::cout << "Unknown parameter: " << name << "\n"; exit(1);
            }
        }
    }

    void ParameterHandler::PrintParametersDifferentFromDefault(std::ostream& out) {
        std::string complete_output = "";
        for (auto& category : categories_) {
            //out << category.name << "\n";
            std::string category_output = "";
            for (auto& parameters : category.parameters) {
                if (parameters.type == "string") {
                    if (parameters_string_[parameters.name].current_value != parameters_string_[parameters.name].default_value) {
                        //out << "\t-" << parameters.name << " = " << parameters_string_[parameters.name].current_value << "\n";
                        category_output += "c \t-" + parameters.name + " = " + parameters_string_[parameters.name].current_value + "\n";
                    }
                } else if (parameters.type == "integer") {
                    if (parameters_integer_[parameters.name].current_value != parameters_integer_[parameters.name].default_value) {
                        //out << "\t-" << parameters.name << " = " << parameters_integer_[parameters.name].current_value << "\n";
                        category_output += "c \t-" + parameters.name + " = " + std::to_string(parameters_integer_[parameters.name].current_value) + "\n";
                    }
                } else if (parameters.type == "Boolean") {
                    if (parameters_boolean_[parameters.name].current_value != parameters_boolean_[parameters.name].default_value) {
                        //out << "\t-" << parameters.name << " = " << parameters_boolean_[parameters.name].current_value << "\n";
                        category_output += "c \t-" + parameters.name + " = " + std::to_string(parameters_boolean_[parameters.name].current_value) + "\n";
                    }
                } else if (parameters.type == "float") {
                    if (parameters_float_[parameters.name].current_value != parameters_float_[parameters.name].default_value) {
                        //out << "\t-" << parameters.name << " = " << parameters_float_[parameters.name].current_value << "\n";
                        category_output += "c \t-" + parameters.name + " = " + std::to_string(parameters_float_[parameters.name].current_value) + "\n";
                    }
                } else {
                    std::cout << "Internal error, undefined type " << parameters.type << "\n";
                    exit(1);
                }
            }

            if (category_output != "") {
                complete_output += category.name + "\n" + category_output;
            }
        }

        if (complete_output == "") {
            out << "c using default parameters\n";
        } else {
            out << "c using default parameters with the following exceptions:\nc " << complete_output << "c ---\n";
        }
    }

    void ParameterHandler::PrintParameterValues(std::ostream& out) {
        for (auto& category : categories_) {
            out << category.name << "\n";
            for (auto& parameters : category.parameters) {
                if (parameters.type == "string") {
                    out << "\t-" << parameters.name << " = " << parameters_string_[parameters.name].current_value << "\n";
                } else if (parameters.type == "integer") {
                    out << "\t-" << parameters.name << " = " << parameters_integer_[parameters.name].current_value << "\n";
                } else if (parameters.type == "Boolean") {
                    out << "\t-" << parameters.name << " = " << parameters_boolean_[parameters.name].current_value << "\n";
                } else if (parameters.type == "float") {
                    out << "\t-" << parameters.name << " = " << parameters_float_[parameters.name].current_value << "\n";
                } else {
                    std::cout << "Internal error, undefined type " << parameters.type << "\n";
                    exit(1);
                }
            }
        }
    }

    void ParameterHandler::PrintHelpSummary(std::ostream& out) {
        if (parameters_string_.empty() && parameters_integer_.empty() && parameters_boolean_.empty() && parameters_float_.empty()) { std::cout << "No parameters declared!\n"; exit(1); }

        for (auto& category : categories_) {
            out << category.name << ". " << category.short_description << "\n";
            for (auto& parameters : category.parameters) {
                if (parameters.type == "string") {
                    auto& entry = parameters_string_[parameters.name];

                    out << "\t-" << entry.name << ".\n\t\tString. " << entry.short_description << "\n";
                    out << "\t\tDefault value: " << entry.default_value << "\n";
                    if (!entry.allowed_values.empty()) {
                        out << "\t\tAllowed values: ";
                        for (size_t i = 0; i < entry.allowed_values.size() - 1; i++) {
                            out << entry.allowed_values[i] << ", ";
                        }
                        out << entry.allowed_values.back() << "\n";
                    }
                    if (entry.optional) out << "Optional" << std::endl;
                } else if (parameters.type == "integer") {
                    auto& entry = parameters_integer_[parameters.name];

                    out << "\t-" << entry.name << ".\n\t\tInteger. " << entry.short_description << "\n";
                    out << "\t\tDefault value: " << entry.default_value << "\n";
                    out << "\t\tRange = [" << entry.min_value << ", " << entry.max_value << "]\n";
                } else if (parameters.type == "Boolean") {
                    auto& entry = parameters_boolean_[parameters.name];

                    out << "\t-" << entry.name << ". Boolean.\n\t\t" << entry.short_description << "\n";
                    out << "\t\tDefault value: " << entry.default_value << "\n";
                } else if (parameters.type == "float") {
                    auto& entry = parameters_float_[parameters.name];

                    out << "\t-" << entry.name << ". Float.\n\t\t" << entry.short_description << "\n";
                    out << "\t\tDefault value: " << entry.default_value << "\n";
                    out << "\t\tRange = [" << entry.min_value << ", " << entry.max_value << "]\n";
                } else {
                    std::cout << "Internal error, undefined type " << parameters.type << "\n";
                    exit(1);
                }
            }
        }
    }

    void ParameterHandler::CheckStringParameter(const std::string& parameter_name, const std::string& value) {
        if (parameters_string_.count(parameter_name) == 0) { std::cout << "Need to define string parameter " << parameter_name << " before it can be set!\n"; exit(1); }
        auto& allowed_values = parameters_string_[parameter_name].allowed_values;
        bool optional = parameters_string_[parameter_name].optional;
        if (optional && value == "") return;
        if (!allowed_values.empty() && std::find(allowed_values.begin(), allowed_values.end(), value) == allowed_values.end()) {
            std::cout << "The passed value " << value << " is not in the list of allowed values for string parameter " << value << "\n";
            std::cout << "Allowed values: ";
            for (size_t i = 0; i < allowed_values.size() - 1; i++) {
                std::cout << allowed_values[i] << ", ";
            }
            std::cout << allowed_values.back() << "\n";
            exit(1);
        }
    }

    void ParameterHandler::CheckIntegerParameter(const std::string& parameter_name, int64_t value) {
        if (parameters_integer_.count(parameter_name) == 0) { std::cout << "Need to define integer parameter " << parameter_name << " before it can be set!\n"; exit(1); }
        if (parameters_integer_[parameter_name].min_value > value || value > parameters_integer_[parameter_name].max_value) {
            std::cout << "The passed value " << value << " is not in the allowed range for integer parameter " << parameter_name << "\n";
            std::cout << "The allowed range is [" << parameters_integer_[parameter_name].min_value << ", " << parameters_integer_[parameter_name].max_value << "]\n";
            exit(1);
        }
    }

    void ParameterHandler::CheckBooleanParameter(const std::string& parameter_name, bool value) {
        if (parameters_boolean_.count(parameter_name) == 0) { std::cout << "Need to define Boolean parameter " << parameter_name << " before it can be set!\n"; exit(1); }
    }

    void ParameterHandler::CheckFloatParameter(const std::string& parameter_name, double value) {
        if (parameters_float_.count(parameter_name) == 0) { std::cout << "Need to define float parameter " << parameter_name << " before it can be set!\n"; exit(1); }
        if (parameters_float_[parameter_name].min_value > value || value > parameters_float_[parameter_name].max_value) {
            std::cout << "The passed value " << value << " is not in the allowed range for float parameter " << parameter_name << "\n";
            std::cout << "The allowed range is [" << parameters_float_[parameter_name].min_value << ", " << parameters_float_[parameter_name].max_value << "]\n";
            exit(1);
        }
    }

    void ParameterHandler::CheckParameters() const {
        if (GetIntegerParameter("max-num-nodes") > (int(1) << GetIntegerParameter("max-depth")) - 1)
        {
            std::cout << "Error: The number of nodes exceeds the limit imposed by the depth!\n";
            exit(1);
        }
    }
}