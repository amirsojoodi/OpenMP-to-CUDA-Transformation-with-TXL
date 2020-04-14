% 3 tests
% Report
% Introduce the problem extensively
% Our contribution and problems
% conclusion and result

include "c.grm"

redefine postfix_extension
	...
    |   [SPOFF] '<'<'< [SPON] [list argument_expression] [SPOFF]'>'>'> '( [SPON] [list argument_expression] 
#ifdef GNU
	    [opt dotdot]
#endif
	')
end redefine

% Main function 
function main
    replace [program]
		P [program]
    export Extras [repeat function_definition_or_declaration]
       _
	%Variable 1: list of IDs
	%Variable 2: List of declarations
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
	    
	construct Def [function_definition_or_declaration]
		'#define BLOCK_SIZE 64
    
	export Extras
    	Extras [. Def]
    
	by
       Extras [. Main] [. Rest]
end function


function doTransform
	replace [block]
		B [block]
    by
		B %[rule that look for array names]
		%[visit every declaration in the block]
		%Get rid of reduntant variables and sort the list

		[resolve_malloc]
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
	
	construct Statement1 [declaration_or_statement]
		'kernel '<'<'<'( '( E2 ') '- '1')'/'BLOCK_SIZE, 'BLOCK_SIZE'>'>'>'( ArrayName ', E2 ')';
	by
		%Pragma
		%ForLoop
		Statement1
		'cudaDeviceSynchronize();
		Rest
end rule

%(rule resolve_pragma_omp_for
	replace [declaration_or_statement]
		'#pragma omp parallel
	by
		'#pragma
end rule)%





