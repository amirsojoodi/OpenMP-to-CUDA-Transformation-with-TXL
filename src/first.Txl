% Parser for C programs
% Jim Cordy, January 2008

% Using Gnu C grammar
include "c.grm"

% Main function 
function main
    replace [program]
		P [program]
    export Extras [repeat function_definition_or_declaration]
       _
    by
    	P [findMainFunction]
		[prependExtras]
end function

function findMainFunction
   replace * [function_definition]
        'int 'main () 
	    B [block]
   by
        'int 'main () 
        B [doTransform]
end function

function prependExtras
    import  Extras [repeat function_definition_or_declaration]
    replace * [repeat function_definition_or_declaration]
        Main [function_definition_or_declaration]
		Rest [repeat function_definition_or_declaration]
    deconstruct Main
        'int 'main () 
	    B [block]
	    
%	construct KernelFuntion [function_definition_or_declaration]
%		'void 'kernel (xxx){
%			some kernel stuff
%		}
	construct Def [function_definition_or_declaration]
		'#define foo 10
    
	export Extras
    	Extras [. Def] % [. KernelFunction] 
%
%	construct xyz [stirnglit]
%		_ [putp Extras]
    by
       Extras [. Main] [. Rest]
end function


function doTransform
	replace [block]
		B [block]
    by
		B [resolve_malloc]
		[resolve_free]
%		[rule 3]
end function

rule resolve_malloc
	replace [expression_statement]
		Id[id] Op[assignment_operator] CastOp[cast_operator] Expr[unary_expression]
	deconstruct * Expr
		'malloc ExpressionExtension[repeat postfix_extension]
	by
		Id Op CastOp 'cudaMallocManaged ExpressionExtension
end rule

rule resolve_free
	replace [macro_call]
		'free(MacroArgs[macro_arguments])
	by
		'cudaFree(MacroArgs)
end rule
