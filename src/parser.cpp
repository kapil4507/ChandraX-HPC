#include "parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

using namespace std;

MatrixDimensions parseXML(const string& xmlPath) {
    MatrixDimensions dims;
    ifstream file(xmlPath);
    if (!file.is_open()) {
        cerr << "Error: Could not open XML file: " << xmlPath << endl;
        return dims;
    }

    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();

    // Regex to find Axis_Array blocks in PDS4
    regex axisArrayRegex(R"(<Axis_Array>([\s\S]*?)</Axis_Array>)");
    regex nameRegex(R"(<axis_name>([^<]+)</axis_name>)");
    regex elementsRegex(R"(<elements>([^<]+)</elements>)");
    regex dataTypeRegex(R"(<data_type>([^<]+)</data_type>)");
    regex fileNameRegex(R"(<file_name>([^<]+)</file_name>)");
    regex echoSamplesRegex(R"(<isda:samples_per_echo_line>([^<]+)</isda:samples_per_echo_line>)");
    regex numPolsRegex(R"(<isda:num_polarizations>([^<]+)</isda:num_polarizations>)");

    // Extract data type first (usually resides inside Element_Array)
    smatch match;
    if (regex_search(content, match, dataTypeRegex)) {
        dims.dataType = match[1].str();
    }

    // Extract binary file name
    if (regex_search(content, match, fileNameRegex)) {
        dims.fileName = match[1].str();
    }

    // Extract samples per echo line
    if (regex_search(content, match, echoSamplesRegex)) {
        dims.samplesPerEcho = stoi(match[1].str());
    } else {
        dims.samplesPerEcho = 1024; // Default fallback
    }

    // Extract number of polarizations
    if (regex_search(content, match, numPolsRegex)) {
        dims.numPols = stoi(match[1].str());
    } else {
        dims.numPols = 2; // Default fallback
    }

    regex centerFreqRegex(R"(<isda:radar_center_frequency unit="Hz">([^<]+)</isda:radar_center_frequency>)");
    regex slantRangeRegex(R"(<isda:slant_range_near_edge unit="m">([^<]+)</isda:slant_range_near_edge>)");
    regex prfRegex(R"(<isda:pulse_repetition_frequency unit="Hz">([^<]+)</isda:pulse_repetition_frequency>)");

    if (regex_search(content, match, centerFreqRegex)) {
        dims.centerFrequency = stod(match[1].str());
    }
    if (regex_search(content, match, slantRangeRegex)) {
        dims.slantRange = stod(match[1].str());
    }
    if (regex_search(content, match, prfRegex)) {
        dims.prf = stod(match[1].str());
    }

    auto words_begin = sregex_iterator(content.begin(), content.end(), axisArrayRegex);
    auto words_end = sregex_iterator();

    for (sregex_iterator i = words_begin; i != words_end; ++i) {
        smatch m = *i;
        string block = m.str();

        smatch nameMatch;
        smatch elementsMatch;

        if (regex_search(block, nameMatch, nameRegex) && 
            regex_search(block, elementsMatch, elementsRegex)) {
            string name = nameMatch[1].str();
            int val = stoi(elementsMatch[1].str());

            if (name == "Line" || name == "Row") {
                dims.lines = val;
            } else if (name == "Sample" || name == "Column") {
                dims.totalLineElements = val;
            }
        }
    }

    return dims;
}
