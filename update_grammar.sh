#!/bin/bash
echo "Updating external grammar..."
cd external/rtamt/rtamt/antlr/grammar/tl
echo "LTL"
antlr4 -visitor -no-listener -o ../../parser/ltl/ -Dlanguage=Python3 LtlLexer.g4
antlr4 -visitor -no-listener -o ../../parser/ltl/ -Dlanguage=Python3 -lib ../../parser/ltl LtlParser.g4
echo "STL"
antlr4 -visitor -no-listener -o ../../parser/stl/ -Dlanguage=Python3 StlLexer.g4
antlr4 -visitor -no-listener -o ../../parser/stl/ -Dlanguage=Python3 -lib ../../parser/stl StlParser.g4

echo "Updating custom grammar..."
cd ../../../../../../crmonitor/rule/fastl/
antlr4 -visitor -no-listener -Dlanguage=Python3 -lib ../../../external/rtamt/rtamt/antlr/grammar/tl FaStlLexer.g4 FaStlParser.g4
cd ../../..
echo "Done."