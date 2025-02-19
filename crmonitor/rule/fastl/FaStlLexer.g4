lexer grammar FaStlLexer ;
import StlLexer ;

// No digits
fragment IdentifierPart
	: ( IdentifierStart | '.' | '/' ) ;

EXIST
    : 'E' ;
FORALL
    : 'A' ;

VEHICLE
    : 'a' ;

AndsmoothOperator
    : 'andsmooth' ;

HistoricallyDurationOperator
	: 'historicallyDuration' ;

HistoricallyDurationSeverityOperator
	: 'historicallyDurationSeverity' ;

// Preserve whitespace
WHITESPACE
	: [ \t\r\u000C]+ -> channel(HIDDEN) ;

IO_TYPE_INPUT
	: '_i';
