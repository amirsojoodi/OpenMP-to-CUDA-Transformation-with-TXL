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
	    
	construct KernelFuntion [function_definition_or_declaration]
		'void 'kernel(){
			;
		}
	construct Def [function_definition_or_declaration]
		'#define foo 10
    
	export Extras
    	Extras [. Def] %[. KernelFunction] 
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
		[resolve_pragma_omp_for]
end function

rule resolve_malloc
	replace [simple_statement]
		Id[id] Op[assignment_operator] CastOp[cast_operator] Expr[unary_expression]
	deconstruct * Expr
		'malloc(OrigArgs[list argument_expression])		
	by
		'cudaMallocManaged(&Id, OrigArgs)
end rule

rule resolve_free
	replace [macro_call]
		'free(MacroArgs[macro_arguments])
	by
		'cudaFree(MacroArgs)
end rule

rule resolve_pragma_omp_for
	replace [repeat declaration_or_statement]
		Pragma [declaration_or_statement]
		ForLoop [declaration_or_statement]
	deconstruct Pragma
		'#pragma omp parallel
	deconstruct ForLoop
		ForStatement[for_statement]
	by
		ForLoop
end rule

%(rule resolve_pragma_omp_for
	replace [declaration_or_statement]
		'#pragma omp parallel
	by
		'#pragma
end rule)%





