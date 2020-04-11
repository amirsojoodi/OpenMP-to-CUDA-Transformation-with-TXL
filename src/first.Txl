% Parser for C programs
% Jim Cordy, January 2008

% Using Gnu C grammar
include "c.grm"

redefine postfix_expression
	...
    |   [SP] '<'<'< [list argument_expression] '>'>'> '( [SPON] [list argument_expression] 
#ifdef GNU
	    [opt dotdot]
#endif
	[SPOFF] ')
end redefine

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
	    
	construct KernelFunction [function_definition_or_declaration]
		'void 'kernel(){
			;
		}
	construct Def [function_definition_or_declaration]
		'#define BLOCK_SIZE 64
    
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
		[add_kernel_function]
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

rule add_kernel_function
	replace $ [repeat declaration_or_statement]
		Pragma [preprocessor]
		ForLoop [for_statement]
		Rest [repeat declaration_or_statement]
	construct PragmaStr [stringlit]
		_ [quote Pragma]
	where
		PragmaStr [grep 'pragma][grep 'omp][grep 'parallel]
	deconstruct ForLoop
		'for (D [decl_specifiers] Init[list init_declarator+] '; Condition[conditional_expression] '; Step[opt expression_list] ) 
			SubStatement[sub_statement]
	%	'for ( [decl_specifiers] [list init_declarator+] '; [opt expression_list] '; [opt expression_list] ) [sub_statement]
	deconstruct * [reference_id] Init
		Var [id] 
	deconstruct Condition
		E1 [shift_expression] Op [relational_operator] E2 [shift_expression]
	deconstruct * [postfix_expression] SubStatement
		ArrayName [id] '[ A [assignment_expression] '] 
	construct KernelFunction [function_definition_or_declaration]
		'__global__ 'void 'kernel('int* ArrayName, 'int 's){
            'int Var '= 'blockIdx.'x '* 'blockDim.'x '+ 'threadIdx.'x;

			'if (Var < 's)
				SubStatement
        }

	import  Extras [repeat function_definition_or_declaration]
	export Extras 
		Extras [. KernelFunction]
	by
		%Pragma
		%ForLoop
		'kernel '<'<'<'( '( E2 ') '- '1')'/'BLOCK_SIZE, 'BLOCK_SIZE'>'>'>'( ArrayName ', E2 ')';
		Rest
end rule

%(rule resolve_pragma_omp_for
	replace [declaration_or_statement]
		'#pragma omp parallel
	by
		'#pragma
end rule)%





