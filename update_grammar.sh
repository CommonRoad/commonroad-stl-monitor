echo "Updating external grammar..."
cd external/rtamt/rtamt/antlr/grammar/tl
java -jar /home/florian/antlr-4.9.3-complete.jar -visitor -no-listener -o ../../parser/stl/ -Dlanguage=Python3 StlLexer.g4 StlParser.g4
echo "Updating custom grammar..."
cd ../../../../../../crmonitor/rule/fastl/
java -jar /home/florian/antlr-4.9.3-complete.jar -visitor -no-listener -Dlanguage=Python3 -lib ../../../external/rtamt/rtamt/antlr/grammar/tl FaStlLexer.g4 FaStlParser.g4
echo "Done."