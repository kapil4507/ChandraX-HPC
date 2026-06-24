#include "parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

MatrixDimensions parseXML(const std::string& xmlPath) {
    MatrixDimensions dims;
    std::ifstream file(xmlPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open XML file: " << xmlPath << std::endl;
        return dims;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Regex to find Axis_Array blocks in PDS4
    std::regex axisArrayRegex(R"(<Axis_Array>([\s\S]*?)</Axis_Array>)");
    std::regex nameRegex(R"(<axis_name>([^<]+)</axis_name>)");
    std::regex elementsRegex(R"(<elements>([^<]+)</elements>)");
    std::regex dataTypeRegex(R"(<data_type>([^<]+)</data_type>)");
    std::regex fileNameRegex(R"(<file_name>([^<]+)</file_name>)");
    std::regex echoSamplesRegex(R"(<isda:samples_per_echo_line>([^<]+)</isda:samples_per_echo_line>)");
    std::regex numPolsRegex(R"(<isda:num_polarizations>([^<]+)</isda:num_polarizations>)");

    // Extract data type first (usually resides inside Element_Array)
    std::smatch match;
    if (std::regex_search(content, match, dataTypeRegex)) {
        dims.dataType = match[1].str();
    }

    // Extract binary file name
    if (std::regex_search(content, match, fileNameRegex)) {
        dims.fileName = match[1].str();
    }

    // Extract samples per echo line
    if (std::regex_search(content, match, echoSamplesRegex)) {
        dims.samplesPerEcho = std::stoi(match[1].str());
    } else {
        dims.samplesPerEcho = 1024; // Default fallback
    }

    // Extract number of polarizations
    if (std::regex_search(content, match, numPolsRegex)) {
        dims.numPols = std::stoi(match[1].str());
    } else {
        dims.numPols = 2; // Default fallback
    }

    auto words_begin = std::sregex_iterator(content.begin(), content.end(), axisArrayRegex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch m = *i;
        std::string block = m.str();

        std::smatch nameMatch;
        std::smatch elementsMatch;

        if (std::regex_search(block, nameMatch, nameRegex) && 
            std::regex_search(block, elementsMatch, elementsRegex)) {
            std::string name = nameMatch[1].str();
            int val = std::stoi(elementsMatch[1].str());

            if (name == "Line" || name == "Row") {
                dims.lines = val;
            } else if (name == "Sample" || name == "Column") {
                dims.totalLineElements = val;
            }
        }
    }

    return dims;
}
