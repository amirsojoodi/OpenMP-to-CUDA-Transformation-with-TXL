% A Simple transformation from OpenMP to CUDA
% Nicolas Merz, AmirHossein Sojoodi, Spring 2020

% Using Gnu C grammar
include "c.grm"

redefine postfix_extension
	...
    |   [SPOFF] '<'<'< [SPON] [list argument_expression] [SPOFF] '>'>'> '( [SPON] [list argument_expression] 
#ifdef GNU
	    [opt dotdot]
#endif
	')
end redefine

define single_decl_no_semi
    [decl_specifiers] [declarator] [opt ',]
end define

% Main function 
function main
    replace [program]
		P [program]
    export Extras [repeat function_definition_or_declaration]
       _
    export Ids [repeat id]
        _
    export ArgExpressions [list argument_expression]
        _ %[list argument_expression]
    export Declarations [list argument_declaration]
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
        B [extract_ids]
        [extract_declarations]
		[resolve_malloc]
		[resolve_free]
		[add_kernel_function]
end function

rule resolve_malloc
	replace $ [simple_statement]
		Id[id] Op[assignment_operator] CastOp[cast_operator] Expr[unary_expression]
    import Ids [repeat id]
    deconstruct * [id] Ids
        Id
	deconstruct * Expr
		'malloc ( OrigArgs [list argument_expression] )		
	by
		'cudaMallocManaged ( '& Id, OrigArgs)
end rule

rule resolve_free
	replace $ [macro_call]
		'free ( MacroArgs [macro_arguments] )
    deconstruct * [id] MacroArgs
        Id [id]
    import Ids [repeat id]
    deconstruct * [id] Ids
        Id
	by
		'cudaFree ( MacroArgs )
end rule

rule extract_ids
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
	by
		Pragma
		ForLoop [extract_id SubStatement]
		Rest
end rule

rule extract_id SubStatement [sub_statement]
    replace $ [repeat declaration_or_statement]
        DoS [repeat declaration_or_statement]
    deconstruct * [postfix_expression] DoS
		ArrayName [id] '[ A [assignment_expression] '] 
    import Ids [repeat id]
    export Ids
        Ids [. ArrayName]
        [remove_duplicates]
    by
        DoS
end rule

rule remove_duplicates
	replace [repeat id]
		Id1[id] Id2[id] RestIds[repeat id]
	where 
		Id1 [= Id2]
	by
		Id1 RestIds
end rule

rule extract_declarations
    replace $ [repeat declaration_or_statement]
        DoS [repeat declaration_or_statement]
    deconstruct * [declaration] DoS
        Spec [decl_specifiers] Decl [declarator] ';
    construct ArgDecl [argument_declaration]
        Spec Decl
        
    deconstruct * [id] Decl
        Id [id]
    import Ids [repeat id]
    deconstruct * [id] Ids
        Id
    construct ArgExpr [list argument_expression]
        Id
    import ArgExpressions [list argument_expression]
    export ArgExpressions
		_ [construct_arg_expressions_1 ArgExpressions ArgExpr]
		[construct_arg_expressions_2 ArgExpressions ArgExpr]			
    import Declarations [list argument_declaration]
    export Declarations
        Declarations [, ArgDecl]
    by
        DoS
end rule

function construct_arg_expressions_2 ArgExpressions[list argument_expression] ArgExpr[list argument_expression]
	replace [list argument_expression]
		%
	deconstruct ArgExpressions
		Expressions [list argument_expression+]
	by
		ArgExpressions[, ArgExpr]
end function

function construct_arg_expressions_1 ArgExpressions[list argument_expression] ArgExpr[list argument_expression]
	replace [list argument_expression]
		%
	deconstruct ArgExpressions
		Empty [empty]
	by
		ArgExpr
end function

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
    import Declarations [list argument_declaration]
    construct Size [argument_declaration]
        'int 's
    construct KernelArgList [list argument_declaration]
        Declarations [, Size]
	construct KernelFunction [function_definition_or_declaration]
		'__global__ 'void 'kernel( KernelArgList ){
            'int Var '= 'blockIdx.'x '* 'blockDim.'x '+ 'threadIdx.'x;

			'if (Var < 's)
				SubStatement
        }

	import  Extras [repeat function_definition_or_declaration]
	export Extras 
		Extras [. KernelFunction]

    import ArgExpressions [list argument_expression]
    construct SizeArg [argument_expression]
        E2
    construct KernelCallArgList [list argument_expression]
        ArgExpressions [, SizeArg]
    construct Statement1 [declaration_or_statement]
		'kernel '<'<'<'( '( E2 ') '- '1')'/'BLOCK_SIZE, 'BLOCK_SIZE'>'>'>'( KernelCallArgList ')';
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





