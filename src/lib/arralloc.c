/*
 * Copyright (c) 2010-2012, The University of Edinburgh
 */

/******************************************************************************
 * Alloc	Interface functions to dynamic store allocators.	      *
 * arralloc()	Allocate rectangular dope-vector (ie using pointers) array    *
 ******************************************************************************/

/*========================== Library include files ===========================*/
#include <stddef.h>
#include <stdarg.h>
#include <stdlib.h>
/*========================== Library declarations ============================*/
/* char	*calloc(); */
/* char	*malloc(); */
/*========================== External function declarations ==================*/
#ifdef	DEBUG
int	malloc_verify();
int	malloc_debug();
#endif
/******************************************************************************
 *  ~arralloc.  Allocate a psuedo array of any dimensionality and type with   *
 *  specified size for each dimension.  Each dimension is	 	      *
 *  an array of pointers, and the actual data is laid out in standard 'c'     *
 *  fashion ie last index varies most rapidly.  All storage is got in one     *
 *  block, so to free whole array, just free the pointer array.               *
 *  array = (double***) arralloc(sizeof(double), 3, 10, 12, 5);		      *
 ******************************************************************************/

/* ALIGN returns the next b byte aligned address after a */
#define ALIGN(a,b)	(int*)( (((long)(a) + (b) - 1)/(b))*(b) )

/* If on an I860 align arrays to cache line boundaries */
#ifdef I860
#define MIN_ALIGN 32
#else
#define MIN_ALIGN 1
#endif

/*----------------------------------------------------------------------*/


void 	subarray(size_t align_size, /* size of object to align the data on */
	 			 size_t size, /* actual size of objects in the array */
				 int ndim,    /* ndim- number of dim left to do */
				 int prdim,   /* prdim - no of pointers in previous iteration */
				 void ***pp,  /* pp - pointer to previous level of the array */
				 void **qq,   /* qq - pointer to start of this level */
				 int *dimp, 
				 int index)
{
   int	*dd = ALIGN(qq,align_size);	/*aligned pointer only used in last recursion*/
   int	**dpp = (int**)pp;
   int i, dim = dimp[index];

   if(ndim > 0)		/* General case - set up pointers to pointers  */
   {
      for( i = 0; i < prdim; i++)
	 pp[i] = qq + i*dim;	/* previous level points to us */

      subarray(align_size, size, ndim-1, prdim*dim,
				(void***)qq,	/* my level filled in next */
				qq+prdim*dim,	/* next level starts later */
				dimp, (index+1) );
   }
   else			/* Last recursion - set up pointers to data   */
      for( i = 0; i < prdim; i++)
	 dpp[i] = dd + (i*dim)*size/sizeof(int);
}
            


/*-----------------------------------------------------------------------*/

/* 
 * if REFS is defined the va macros are dummied out. This is because the
 * GreenHills va_arg macro will not get past the cref utility.
 * This way the call tree can still be constructed. Do NOT under
 * any circumstance define REFS when compiling the code.
 */

#if REFS
   #undef va_start
   #undef va_arg
   #undef va_end
   #undef va_list
   #define va_list int
   #define va_start( A , B) ( A = (int) B)
   #define va_end( A ) ( A = 0 )
   #define va_arg( A , T ) ( A = (T) 0)
#endif

void *arralloc(size_t size, int ndim, ...)
{
   va_list	ap;
   void		**p, **start;
   int		idim;
   long		n_ptr = 0, n_data = 1;
   int 		*dimp;
   size_t	align_size;

   va_start(ap, ndim);

   if( size % sizeof(int) != 0 )  /* Code only works for 'word' objects */
      return 0;
   /* we want to align on both size and MIN_ALIGN */
   if( size > MIN_ALIGN )
   {
   	align_size = size;
   }
   else
   {
   	align_size = MIN_ALIGN;
   }
   while( (align_size % size) || (align_size % MIN_ALIGN) )
   {
   	align_size++;
   }
   /*
    * Cycle over dims,  accumulate # pointers & data items.
    */
   if ( NULL == (dimp=(int *)malloc( ndim * sizeof(int) )))
	return 0;

   for(idim = 0; idim < ndim; idim++)
   {
      dimp[idim] = va_arg(ap, int);
      n_data *= dimp[idim];
      if( idim < ndim-1 )
	 n_ptr  += n_data;
   }
   va_end(ap);   


   /*
    *  Allocate space  for pointers and data.
    */
   if( (start = (void**)malloc(
		(size_t)((n_data*size)+align_size+(n_ptr*sizeof(void**))))) == 0)
      return 0;
   /*
    * Set up pointers to form dope-vector array.
    */
   subarray(align_size, size, ndim-1, 1, &p, start, dimp, 0);
   free( dimp );

   return (void*)p;
}


