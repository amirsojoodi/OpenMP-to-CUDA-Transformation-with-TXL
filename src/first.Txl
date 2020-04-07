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
    by
       Extras [. Main] [. Rest]
end function


function doTransform
   replace [block]
      B [block]
    by
      B [rule 1]
        [rule 2]
	[rule 3]
end function



    import  Extras [repeat function_definition_or_declaration]
    construct KernelFuntion [function_definition_or_declaration]
       'void 'kernel (xxx){
          some kernel stuff
       }
    construct Def [function_definition_or_decalaration]
       '#define foo 10
    export Extras
       Extras [. KernelFunction] [. Def]


    contruct xyz [stirnglit]
       _ [putp Extras]
