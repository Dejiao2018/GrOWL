/*
 * Copyright 2013, M. Bogdan, E. van den Berg, W. Su, and E.J. Candes
 *
 * This file is part of SLOPE Toolbox version 1.0.
 *
 *   The SLOPE Toolbox is free software: you can redistribute it
 *   and/or  modify it under the terms of the GNU General Public License
 *   as published by the Free Software Foundation, either version 3 of
 *   the License, or (at your option) any later version.
 *
 *   The SLOPE Toolbox is distributed in the hope that it will
 *   be useful, but WITHOUT ANY WARRANTY; without even the implied
 *   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *   See the GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with the SLOPE Toolbox. If not, see
 *   <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
// #include <Python.h>
#include "proxSortedL1.h"


/* ----------------------------------------------------------------------- */
float* evaluateProx(float *y, float *lambda, size_t n, int *order)
/* ----------------------------------------------------------------------- */
{  float   d;
   float  *s     = NULL;
   float  *w     = NULL;
   float *x      = NULL;
   size_t  *idx_i = NULL;
   size_t *idx_j = NULL;
   size_t  i,j,k;

   
   // size_t      result = 0;
   
   /* Allocate memory */
  
   s     = (float *)malloc(sizeof(float) * n);
   w     = (float *)malloc(sizeof(float) * n);
   x     = (float *)malloc(sizeof(float) * n);
   idx_i = (size_t *)malloc(sizeof(size_t) * n);
   idx_j = (size_t *)malloc(sizeof(size_t) * n);
   
   if ((s != NULL) && (w != NULL) && (idx_i != NULL) && (idx_j != NULL))
   {
      k = 0;
      for (i = 0; i < n; i++)
      {
         idx_i[k] = i;
         idx_j[k] = i;
         s[k]     = y[i] - lambda[i];
         w[k]     = s[k];
         
         while ((k > 0) && (w[k-1] <= w[k]))
         {  k --;
            idx_j[k] = i;
            s[k]    += s[k+1];
            w[k]     = s[k] / (i - idx_i[k] + 1);
         }
         
         k++;
      }
      
      if (order == NULL)
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[i] = d;
            }
         }
      }
      else
      {  for (j = 0; j < k; j++)
         {  d = w[j]; if (d < 0) d = 0;
            for (i = idx_i[j]; i <= idx_j[j]; i++)
            {  x[order[i]] = d;
            }
         }
      }
   }
   else
   {  
      return x;
   }
   
   /* Deallocate memory */
   if (s     != NULL) free(s);
   if (w     != NULL) free(w);
   if (idx_i != NULL) free(idx_i);
   if (idx_j != NULL) free(idx_j);
   
   return x;
}


void free_mem(float *a)
{
   free(a);
   // printf("memory freed");
}
