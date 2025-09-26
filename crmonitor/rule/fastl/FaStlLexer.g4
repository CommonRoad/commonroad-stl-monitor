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

SigmoidOperator
    : 'sigmoid' ;

HistoricallyDurationOperator
	: 'historicallyDuration' ;

HistoricallyDurationSeverityOperator
	: 'historicallyDurationSeverity' ;

SumIfPositiveOperator
	: 'sum_if_positive' ;

CompareToThresholdScaledOperator
	: 'compare_to_threshold_scaled' ;

ExistsMultipleOperator
	: 'exists_multiple' ;

// Preserve whitespace
WHITESPACE
	: [ \t\r\u000C]+ -> channel(HIDDEN) ;

IO_TYPE
	: ('_i' | '_x');
