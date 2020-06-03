/*
 * Simplified simulation of life evolution
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2019/2020
 *
 * v1.5
 *
 * CHANGES:
 * 1) Float values have been substituted by fixed point arithmetics 
 *	using integers. To simplify, the fixed point arithmetics are done 
 *	with PRECISION in base 10. See precision constant in int_float.h
 * 2) It uses a portable approximation to trigonometric functions using
 *	Taylor polynomials. 
 * 3) nrand48 function has been extracted from glibc source code and 
 *	its internal API simplified to allow its use in the GPU.
 *
 * (c) 2020, Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <cstring>
#include<math.h>
#include<stdbool.h>
#include<cputils.h>
#include<cuda.h>
#include<int_float.h>

/* 
 * Constants: Converted to fixed point with the given PRECISION
 */
#define ENERGY_NEEDED_TO_LIVE		PRECISION / 10	// Equivalent to 0.1
#define ENERGY_NEEDED_TO_MOVE		PRECISION	// Equivalent to 1.0
#define ENERGY_SPENT_TO_LIVE		PRECISION / 5	// Equivalent to 0.2
#define ENERGY_SPENT_TO_MOVE		PRECISION	// Equivalent to 1.0
#define ENERGY_NEEDED_TO_SPLIT		PRECISION * 20	// Equivalent to 20.0

__device__ __constant__ int columnsDev;
/* Structure to store data of a cell */
typedef struct {
	int pos_row, pos_col;		// Position
	int mov_row, mov_col;		// Direction of movement
	int choose_mov[3];		// Genes: Probabilities of 0 turning-left; 1 advance; 2 turning-right
	int storage;			// Food/Energy stored
	int age;			// Number of steps that the cell has been alive
	unsigned short random_seq[3];	// Status value of its particular random sequence
	bool alive;			// Flag indicating if the cell is still alive
} Cell;


/* Structure for simulation statistics */
typedef struct {
	int history_total_cells;	// Accumulated number of cells created
	int history_dead_cells;		// Accumulated number of dead cells
	int history_max_alive_cells;	// Maximum number of cells alive in a step
	int history_max_new_cells;	// Maximum number of cells created in a step
	int history_max_dead_cells;	// Maximum number of cells died in a step
	int history_max_age;		// Maximum age achieved by a cell
	int history_max_food;		// Maximum food level in a position of the culture
} Statistics;


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 *	USE THIS SPACE FOR YOUR KERNEL OR DEVICE FUNTIONS
 *
 */

#include "taylor_trig.h"
#include "glibc_nrand48.h"

/*
 * Get an uniformly distributed random number between 0 and max
 * It uses glibc_nrand, that returns a number between 0 and 2^31
 */
#define int_urand48( max, seq )	(int)( (long)(max) * glibc_nrand48( seq ) / 2147483648 )

/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be modified by the students if needed
 *
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]

/*
 * Function: Choose a new direction of movement for a cell
 * 	This function can be changed and/or optimized by the students
 */
__device__ void cell_new_direction( Cell *cell ) {
	int angle = int_urand48( INT_2PI, cell->random_seq );
	cell->mov_row = taylor_sin( angle );
	cell->mov_col = taylor_cos( angle );
}

void cell_new_direction2( Cell *cell ) {
	int angle = int_urand48( INT_2PI, cell->random_seq );
	cell->mov_row = taylor_sin( angle );
	cell->mov_col = taylor_cos( angle );
}

/*
 * Function: Mutation of the movement genes on a new cell
 * 	This function can be changed and/or optimized by the students
 */

//poner __device__ delante y asi se le puede llamar desde una función __global__
__device__ void cell_mutation( Cell *cell ) {
	/* 1. Select which genes change:
	 	0 Left grows taking part of the Advance part
	 	1 Advance grows taking part of the Left part
	 	2 Advance grows taking part of the Right part
	 	3 Right grows taking part of the Advance part
	*/
	int mutation_type = int_urand48( 4, cell->random_seq );
	/* 2. Select the amount of mutation (up to 50%) */
	int mutation_percentage = int_urand48( PRECISION / 2, cell->random_seq );
	/* 3. Apply the mutation */
	int mutation_value;
	switch( mutation_type ) {
		case 0:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[0] += mutation_value;
			break;
		case 1:
			mutation_value = intfloatMult( cell->choose_mov[0] , mutation_percentage );
			cell->choose_mov[0] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 2:
			mutation_value = intfloatMult( cell->choose_mov[2] , mutation_percentage );
			cell->choose_mov[2] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 3:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[2] += mutation_value;
			break;
	}
	/* 4. Correct potential precision problems */
	cell->choose_mov[2] = PRECISION - cell->choose_mov[1] - cell->choose_mov[0];
}

/*
 * CUDA block reduction
 * Inputs: 
 *	Device pointer to an array of int of any size
 *	Size of the array
 *	Device pointer to an int to store the result
 * 
 * Launching parameters:
 *	One-dimesional grid of any size
 *	Any valid block size
 *	Dynamic shared memory size equal to: sizeof(int) * block size
 *
 * (c) 2020, Arturo Gonzalez-Escribano
 * Simplification for an assignment in a Parallel Computing course,
 * Computing Engineering Degree, Universidad de Valladolid
 * Academic year 2019/2020
 */

/*
__global__ Cell* myrealloc(int oldsize, int newsize, Cell* old)
{
    Cell* newT = (Cell*) malloc (newsize*sizeof(Cell));

    int i;

    for(i=0; i<oldsize; i++)
    {
        newT[i] = old[i];
    }

    free(old);
    return newT;
}*/

__global__ void reductionMax(int* array, int size, int* result)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ int buffer[ ];
	if ( globalPos < size ) { 
		array[globalPos] -= array[globalPos] / 20;
		buffer[ threadIdx.x ] = array[ globalPos ];
	}
	else buffer[ threadIdx.x ] = 0.0f;
	__syncthreads();

	for( int step=blockDim.x/2; step>=1; step /= 2 ) {
		if ( threadIdx.x < step )
			if ( buffer[ threadIdx.x ] < buffer[ threadIdx.x + step ] )
				buffer[ threadIdx.x ] = buffer[ threadIdx.x + step ];
		if ( step > 32 )
			__syncthreads();
	}

	if ( threadIdx.x == 0 )
		atomicMax( result, buffer[0] );
}

__device__ void reductionMax2(Cell* array, int size, int *result)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ int buffer[ ];
	if ( globalPos < size ) { 
		buffer[ threadIdx.x ] = array[ globalPos ].age;
	}
	else buffer[ threadIdx.x ] = 0.0f;
	__syncthreads();

	for( int step=blockDim.x/2; step>=1; step /= 2 ) {
		if ( threadIdx.x < step )
			if ( buffer[ threadIdx.x ] < buffer[ threadIdx.x + step ] )
				buffer[ threadIdx.x ] = buffer[ threadIdx.x + step ];
		if ( step > 32 )
			__syncthreads();
	}

	if ( threadIdx.x == 0 )
		atomicMax( result, buffer[0] );
}

__global__ void inicializarDevices(int* array, int* array2, int* array3, int* array4)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos==0){
		array[globalPos] = 0;
		array2[globalPos] = 0;
		array3[globalPos] = 0;
		array4[globalPos] = 0;
	}
}

__global__ void iniciarCultureCells(int* array, int size)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos<size){
		array [globalPos] = 0;
	}
}

__global__ void comerComida(int* array, Cell* cells, int* size, int rows, int columns)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < size[0] ) {
		if ( cells[globalPos].alive ) {
			int posicion = (cells[globalPos].pos_row / PRECISION) * columns + (cells[globalPos].pos_col / PRECISION);
			array[posicion] = 0;
		}
	}
}

__global__ void limpiarCelulas(Cell* cells2, int* numCells, int* alive, Cell* cellsSal)
{	
	int pos = 0;
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < numCells[0] ) {
		if ( cells2[globalPos].alive ) {
			pos = atomicAdd(alive,1);
			cellsSal[pos] = cells2[globalPos];
		}
	}
}

__global__ void limpiarCelulas2(Cell* cellsEntrada, int* alive, Cell* cellsSalida, int* numCells)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	int p = alive[0];
	if ( globalPos < p ) {
		cellsSalida[globalPos] = cellsEntrada[globalPos];
	}
	if(globalPos==0){
		numCells[0] = alive[0];
	}
}

__global__ void anyadirCelulas(int* size, Cell* newCells, Cell* cellsSal)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < size[0] ) {
		cellsSal[ globalPos ] = newCells[ globalPos ];
	}	
}

__global__ void anyadirCelulas2(Cell* cellsEntrada, int* stepNewCells, int* numCells, Cell* cellsSalida)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < stepNewCells[0]){
		cellsSalida[numCells[0] + globalPos] = cellsEntrada[globalPos];
		//printf("esta es la salida %i en la posicion %i\n",cellsSalida[numCells + globalPos].age, numCells + globalPos);
	}	
}

__global__ void final(int* cDevice, int* stepNewCellsDevice, int* stepDeadCellsDevice, int* numAliveDevice, Statistics* stats)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos==0){
		if ( cDevice[0] > stats[0].history_max_food ) stats[0].history_max_food = cDevice[0];
		// Statistics: Max new cells per step
		if ( stepNewCellsDevice[0] > stats[0].history_max_new_cells ) stats[0].history_max_new_cells = stepNewCellsDevice[0];
		// Statistics: Accumulated dead and Max dead cells per step
		stats[0].history_dead_cells += stepDeadCellsDevice[0];
		if ( stepDeadCellsDevice[0] > stats[0].history_max_dead_cells ) stats[0].history_max_dead_cells = stepDeadCellsDevice[0];
		// Statistics: Max alive cells per step
		if ( numAliveDevice[0] > stats[0].history_max_alive_cells ) stats[0].history_max_alive_cells = numAliveDevice[0];
	}	
}


__global__ void actualizarNumCells(int* numCells, int* stepNewCells)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos==0){
		numCells[0] += stepNewCells[0];
	}	
}

/*
__global__ void actualizarStats(int* history, Statistics* stats)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
		
}*/

//crear una funcion global que inicialice en cada iteracion los valores que tengan que ser nuevamente 0, por ejemplo aDevice, stepDeadCellsDevice ... 
__global__ void movimientoCelulas(Cell* cellsEntrada, int rows, int columns, int* aliveDevice, int* stepDeadCellsDevice, Statistics* s, int* numCells, int* arrayCul, 
	int* arrayCulCells/*, int* foodShareDevice*/)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < numCells[0]){
		if ( cellsEntrada[globalPos].alive ) {
			cellsEntrada[globalPos].age ++;
			//if ( cellsEntrada[globalPos].age > s[0].history_max_age ) s[0].history_max_age = cellsEntrada[globalPos].age; //pasar history_max_age
			if ( cellsEntrada[globalPos].storage < ENERGY_NEEDED_TO_LIVE ) {
				cellsEntrada[globalPos].alive = false;
				atomicSub(aliveDevice,1);
				atomicAdd(stepDeadCellsDevice,1);
			}
			if ( cellsEntrada[globalPos].alive ) {
				if ( cellsEntrada[globalPos].storage < ENERGY_NEEDED_TO_MOVE ) {
					cellsEntrada[globalPos].storage -= ENERGY_SPENT_TO_LIVE;
				}
				else {
					cellsEntrada[globalPos].storage -= ENERGY_SPENT_TO_MOVE;
							
					int prob = int_urand48( PRECISION, cellsEntrada[globalPos].random_seq );
					if ( prob < cellsEntrada[globalPos].choose_mov[0] ) {
						int tmp = cellsEntrada[globalPos].mov_col;
						cellsEntrada[globalPos].mov_col = cellsEntrada[globalPos].mov_row;
						cellsEntrada[globalPos].mov_row = -tmp;
					}
					else if ( prob >= cellsEntrada[globalPos].choose_mov[0] + cellsEntrada[globalPos].choose_mov[1] ) {
						int tmp = cellsEntrada[globalPos].mov_row;
						cellsEntrada[globalPos].mov_row = cellsEntrada[globalPos].mov_col;
						cellsEntrada[globalPos].mov_col = -tmp;
					}
						
					cellsEntrada[globalPos].pos_row += cellsEntrada[globalPos].mov_row;
					cellsEntrada[globalPos].pos_col += cellsEntrada[globalPos].mov_col;

					if ( cellsEntrada[globalPos].pos_row < 0 ) cellsEntrada[globalPos].pos_row += rows * PRECISION;
					if ( cellsEntrada[globalPos].pos_row >= rows * PRECISION) cellsEntrada[globalPos].pos_row -= rows * PRECISION;
					if ( cellsEntrada[globalPos].pos_col < 0 ) cellsEntrada[globalPos].pos_col += columns * PRECISION;
					if ( cellsEntrada[globalPos].pos_col >= columns * PRECISION) cellsEntrada[globalPos].pos_col -= columns * PRECISION;
				}
				int posicion = (cellsEntrada[globalPos].pos_row / PRECISION) * columns + (cellsEntrada[globalPos].pos_col / PRECISION);
				atomicAdd(&arrayCulCells[posicion],1);
				//foodShareDevice[globalPos] = arrayCul[posicion];
			}
		}
	}
	reductionMax2(cellsEntrada, numCells[0], &s[0].history_max_age);
}

/*
__global__ void movimientoCelulas2(Cell* cellsEntrada,  int* numCells, int columns)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < numCells[0]){
		if(cellsEntrada[globalPos].alive){
			
		}
	}
}*/



__global__ void nacimientoCelulas(Cell* cellsEntrada, int* arrayCulCells, int* arrayCul/*, int* foodShareDevice*/, Cell* newCellsDevice, int* aliveDevice, int* stepNewCellsDevice, 
	int* banderaDevice, int* numCells, int columns, Statistics* stats)
{
	int pos = 0;
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < numCells[0]){
		if(cellsEntrada[globalPos].alive){
			int posicion = (cellsEntrada[globalPos].pos_row / PRECISION) * columns + (cellsEntrada[globalPos].pos_col / PRECISION);
			int food = arrayCul[posicion];
			int count = arrayCulCells[posicion];
			int my_food = food / count;
			cellsEntrada[globalPos].storage += my_food;

			if ( cellsEntrada[globalPos].age > 30 && cellsEntrada[globalPos].storage > ENERGY_NEEDED_TO_SPLIT ) {
				atomicAdd(aliveDevice,1);
				pos = atomicAdd(stepNewCellsDevice,1);	//faltaaaaaaaaaaaaaaaaaa
				atomicAdd(&stats[0].history_total_cells,1);		//faltaaaaaaaaaaaaaaa
				//printf("banderaDevice %i\n", banderaDevice[0]);	
				newCellsDevice[ pos ] = cellsEntrada[globalPos];

					// Split energy stored and update age in both cells
				cellsEntrada[globalPos].storage /= 2;
				newCellsDevice[ pos ].storage /= 2;
				cellsEntrada[globalPos].age = 1;
				newCellsDevice[ pos ].age = 1;

					// Random seed for the new cell, obtained using the parent random sequence
				newCellsDevice[ pos ].random_seq[0] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				newCellsDevice[ pos ].random_seq[1] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				newCellsDevice[ pos ].random_seq[2] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				
				cell_new_direction( &cellsEntrada[globalPos] );
				cell_new_direction( &newCellsDevice[ pos ] );
				
				// Mutations of the movement genes in both cells
				cell_mutation( &cellsEntrada[globalPos] );
				cell_mutation( &newCellsDevice[ pos ] );
			}
		}
	} // End cell actions
}


/*
__global__ void limpiarCelulas(Cell* cells, int size, int* alive2, int* posi)
{	
	__shared__ int alive;
	alive = 0;
	__shared__ int pos;
	pos = 0;
	int p = 0;
	//int pos = 0;
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < size ) {
		if ( cells[globalPos].alive ) {
			atomicAdd(&alive,1);
			if ( p != globalPos ) {
				cells[p] = cells[globalPos];
			}
			p = atomicAdd(&pos,1);

		}
	}
	__syncthreads();
	if(globalPos==0)
	alive2[0] = alive;
}*/

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, int *culture, int num_cells, Cell *cells, int num_cells_alive, Statistics sim_stat ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;

	printf("Iteration: %d\n", iteration );
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( culture, i, j ) >= 20 * PRECISION ) symbol = '+';
			else if ( accessMat( culture, i, j ) >= 10 * PRECISION ) symbol = '*';
			else if ( accessMat( culture, i, j ) >= 5 * PRECISION ) symbol = '.';
			else symbol = ' ';

			int t;
			int counter = 0;
			for( t=0; t<num_cells; t++ ) {
				int row = (int)(cells[t].pos_row / PRECISION);
				int col = (int)(cells[t].pos_col / PRECISION);
				if ( cells[t].alive && row == i && col == j ) {
					counter ++;
				}
			}
			if ( counter > 9 ) printf("(M)" );
			else if ( counter > 0 ) printf("(%1d)", counter );
			else printf(" %c ", symbol );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	printf("Num_cells_alive: %04d\nHistory( Cells: %04d, Dead: %04d, Max.alive: %04d, Max.new: %04d, Max.dead: %04d, Max.age: %04d, Max.food: %6f )\n\n", 
		num_cells_alive, 
		sim_stat.history_total_cells, 
		sim_stat.history_dead_cells, 
		sim_stat.history_max_alive_cells, 
		sim_stat.history_max_new_cells, 
		sim_stat.history_max_dead_cells, 
		sim_stat.history_max_age,
		(float)sim_stat.history_max_food / PRECISION
	);
}
#endif

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <maxIter> <max_food> <food_density> <food_level> <short_rnd1> <short_rnd2> <short_rnd3> <num_cells>\n");
	fprintf(stderr,"\tOptional arguments for special food spot: [ <row> <col> <size_rows> <size_cols> <density> <level> ]\n");
	fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j;

	// Simulation data
	int max_iter;			// Maximum number of simulation steps
	int rows, columns;		// Cultivation area sizes
	int *culture;			// Cultivation area values
	int *culture_cells;		// Ancillary structure to count the number of cells in a culture space

	float max_food;			// Maximum level of food on any position
	float food_density;		// Number of food sources introduced per step
	float food_level;		// Maximum number of food level in a new source

	bool food_spot_active = false;	// Special food spot: Active
	int food_spot_row = 0;		// Special food spot: Initial row
	int food_spot_col = 0;		// Special food spot: Initial row
	int food_spot_size_rows = 0;	// Special food spot: Rows size
	int food_spot_size_cols = 0;	// Special food spot: Cols size
	float food_spot_density = 0.0f;	// Special food spot: Food density
	float food_spot_level = 0.0f;	// Special food spot: Food level

	unsigned short init_random_seq[3];	// Status of the init random sequence
	unsigned short food_random_seq[3];	// Status of the food random sequence
	unsigned short food_spot_random_seq[3];	// Status of the special food spot random sequence

	int	num_cells;		// Number of cells currently stored in the list
	Cell	*cells;			// List to store cells information

	// Statistics
	Statistics sim_stat;	
	sim_stat.history_total_cells = 0;
	sim_stat.history_dead_cells = 0;
	sim_stat.history_max_alive_cells = 0;
	sim_stat.history_max_new_cells = 0;
	sim_stat.history_max_dead_cells = 0;
	sim_stat.history_max_age = 0;
	sim_stat.history_max_food = 0.0f;

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read culture sizes, maximum number of iterations */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	max_iter = atoi( argv[3] );

	/* 1.3. Food data */
	max_food = atof( argv[4] );
	food_density = atof( argv[5] );
	food_level = atof( argv[6] );

	/* 1.4. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		init_random_seq[i] = (unsigned short)atoi( argv[7+i] );
	}

	/* 1.5. Read number of cells */
	num_cells = atoi( argv[10] );
	int num_cellsAux2 = 2000000;
	if(num_cells > num_cellsAux2){
		num_cellsAux2 = num_cells;
	}
	num_cellsAux2 = num_cellsAux2*10;
	//num_cellsAux2 = num_cellsAux2*2;

	/* 1.6. Read special food spot */
	if (argc > 11 ) {
		if ( argc < 17 ) {
			fprintf(stderr, "-- Error in number of special-food-spot arguments in the command line\n\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		else {
			food_spot_active = true;
			food_spot_row = atoi( argv[11] );
			food_spot_col = atoi( argv[12] );
			food_spot_size_rows = atoi( argv[13] );
			food_spot_size_cols = atoi( argv[14] );
			food_spot_density = atof( argv[15] );
			food_spot_level = atof( argv[16] );

			// Check non-used trailing arguments
			if ( argc > 17 ) {
				fprintf(stderr, "-- Error: too many arguments in the command line\n\n");
				show_usage( argv[0] );
				exit( EXIT_FAILURE );
			}
		}
	}

#ifdef DEBUG
	/* 1.7. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
	printf("Arguments, Max.food: %f, Food density: %f, Food level: %f\n", max_food, food_density, food_level);
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", init_random_seq[0], init_random_seq[1], init_random_seq[2]);
	if ( food_spot_active ) {
		printf("Arguments, Food_spot, pos(%d,%d), size(%d,%d), Density: %f, Level: %f\n",
			food_spot_row, food_spot_col, food_spot_size_rows, food_spot_size_cols, food_spot_density, food_spot_level );
	}
	printf("Initial cells: %d\n", num_cells );
#endif // DEBUG


	/* 1.8. Initialize random sequences for food dropping */
	for( i=0; i<3; i++ ) {
		food_random_seq[i] = (unsigned short)glibc_nrand48( init_random_seq );
		food_spot_random_seq[i] = (unsigned short)glibc_nrand48( init_random_seq );
	}

	/* 1.9. Initialize random sequences of cells */
	cells = (Cell *)malloc( sizeof(Cell) * (size_t)num_cellsAux2 );
	Cell* cells2 = (Cell *)malloc( sizeof(Cell) * (size_t)num_cellsAux2 );
	if ( cells == NULL ) {
		fprintf(stderr,"-- Error allocating: %d cells\n", num_cells );
		exit( EXIT_FAILURE );
	}
	for( i=0; i<num_cells; i++ ) {
		// Initialize the cell ramdom sequences
		for( j=0; j<3; j++ ) 
			cells[i].random_seq[j] = (unsigned short)glibc_nrand48( init_random_seq );
	}


#ifdef DEBUG
	/* 1.10. Print random seed of the initial cells */
	/*
	printf("Initial cells random seeds: %d\n", num_cells );
	for( i=0; i<num_cells; i++ )
		printf("\tCell %d, Random seq: %hu,%hu,%hu\n", i, cells[i].random_seq[0], cells[i].random_seq[1], cells[i].random_seq[2] );
	*/
#endif // DEBUG


	// CUDA start
	cudaSetDevice(0);
	cudaDeviceSynchronize();

	/* 2. Start global timer */
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

#include "cuda_check.h"


	int tamanoBloq=1024; 
	int numeroBloq;
	int numeroBloq2=0;
	numeroBloq=(rows*columns)/tamanoBloq;
    if ((rows*columns)%tamanoBloq!=0) 
    {
        numeroBloq++;
    }
    /*
    if(num_cellsAux2 > (rows*columns)){
    	numeroBloq = 0;
    	numeroBloq=num_cellsAux2/tamanoBloq;
    	if (num_cellsAux2%tamanoBloq!=0) 
    	{
        numeroBloq++;
    	}
    }*/
	/* 3. Initialize culture surface and initial cells */
	int *maxFoodTemp;
	int *aliveList;
	int *stepDeadCells;
	int *stepNewCells;
	int *numAlive;
	//int *foodShare;
	int *bandera2;
	//int *historyTotalCells;	
	int *numeroCelulas;
	int *bandera;
	Statistics *stats;
	//Statistics *stats2;

	maxFoodTemp = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	numeroCelulas = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	bandera2 = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	//historyTotalCells = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	numAlive = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	stepDeadCells = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	stepNewCells = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	//foodShare = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	aliveList = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	bandera = (int *)malloc( sizeof(int) * (size_t)1 * (size_t)1 );
	culture = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	culture_cells = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );


	stats = (Statistics *)malloc( sizeof(Statistics) * (size_t)1 * (size_t)1 );

	if ( culture == NULL || culture_cells == NULL ) {
		fprintf(stderr,"-- Error allocating culture structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}
	int* pDevice;
	int* p2Device;
	int* cDevice;
	Cell* cellsDevice;
	Cell* newCellsDevice;
	Cell* cellsDevice2;
	int* aDevice;
	int* numeroCelulasDevice;
	//int* foodShareDevice;
	int* numAliveDevice;
	int* stepDeadCellsDevice;
	int* stepNewCellsDevice;
	int* historyMaxAgeDevice;
	int* historyTotalCellsDevice;
	int* banderaDevice;

	Statistics* statsDevice;

	//no hay realloc hacer cudaFree y despues cudaMalloc
	//cudaMalloc((void **) &foodShareDevice, sizeof(int) * (size_t)num_cellsAux2); //max_food
	cudaMalloc((void **) &numAliveDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &numeroCelulasDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &historyMaxAgeDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &historyTotalCellsDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &stepDeadCellsDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &stepNewCellsDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &pDevice, sizeof(int) * (size_t)rows * (size_t)columns); //culture
	cudaMalloc((void **) &p2Device, sizeof(int) * (size_t)rows * (size_t)columns); //culture_cells
	cudaMalloc((void **) &cDevice, sizeof(int) * (size_t)1 * (size_t)1); //max_food
	cudaMalloc((void **) &aDevice, sizeof(int) * (size_t)1 * (size_t)1); //alive
	//cudaMalloc((void **) &banderaDevice, sizeof(int) * (size_t)1 * (size_t)1); //alive
	cudaMalloc((void **) &cellsDevice, sizeof(Cell) * (size_t)num_cellsAux2); //cells entrada
	cudaMalloc((void **) &statsDevice, sizeof(Statistics)); //cells entrada
	cudaMalloc((void **) &cellsDevice2, sizeof(Cell) * (size_t)num_cellsAux2); //cells entrada	
	cudaMalloc((void **) &newCellsDevice, sizeof(Cell) * (size_t)num_cellsAux2); //cells new cells

	stats[0] = sim_stat;

	/*cudaMemcpy( pDevice, culture, sizeof(int) * (size_t)rows * (size_t)columns ,cudaMemcpyHostToDevice );

	prueba2<<<numeroBloq, tamanoBloq>>>( pDevice );

	cudaMemcpy( pDevice, culture, sizeof(int) * (size_t)rows * (size_t)columns ,cudaMemcpyDeviceToHost );*/

	for( i=0; i<rows; i++ )
		for( j=0; j<columns; j++ ) 
			accessMat( culture, i, j ) = 0;

	for( i=0; i<num_cells; i++ ) {
		cells[i].alive = true;
		// Initial age: Between 1 and 20 
		cells[i].age = 1 + int_urand48( 19, cells[i].random_seq );
		// Initial storage: Between 10 and 20 units
		cells[i].storage = 10 * PRECISION + int_urand48( 10 * PRECISION, cells[i].random_seq );
		// Initial position: Anywhere in the culture arena
		cells[i].pos_row = int_urand48( rows * PRECISION, cells[i].random_seq );
		cells[i].pos_col = int_urand48( columns * PRECISION, cells[i].random_seq );
		// Movement direction: Unity vector in a random direction
		cell_new_direction2( &cells[i] );
		// Movement genes: Probabilities of advancing or changing direction: The sum should be 1.00
		cells[i].choose_mov[0] = PRECISION / 3;
		cells[i].choose_mov[2] = PRECISION / 3;
		cells[i].choose_mov[1] = PRECISION - cells[i].choose_mov[0] - cells[i].choose_mov[2];
	}

	// Statistics: Initialize total number of cells, and max. alive
	sim_stat.history_total_cells = num_cells;
	sim_stat.history_max_alive_cells = num_cells;

#ifdef DEBUG
	/* Show initial cells data */
	printf("Initial cells data: %d\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("\tCell %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i, 
				(float)cells[i].pos_row / PRECISION, 
				(float)cells[i].pos_col / PRECISION, 
				(float)cells[i].mov_row / PRECISION, 
				(float)cells[i].mov_col / PRECISION, 
				(float)cells[i].choose_mov[0] / PRECISION, 
				(float)cells[i].choose_mov[1] / PRECISION, 
				(float)cells[i].choose_mov[2] / PRECISION, 
				(float)cells[i].storage / PRECISION,
				cells[i].age );
	}
#endif // DEBUG



	/* 4. Simulation */
	int current_max_food = 0;
	int num_cells_alive = num_cells;
	int iter;
	int max_food_int = max_food * PRECISION;
	numeroBloq2 = num_cells/tamanoBloq;
	if (num_cells%tamanoBloq!=0) 
    {
        numeroBloq2++;
    }
	numeroCelulas[0] = num_cells;
	numAlive[0] = num_cells_alive;
	stats[0].history_total_cells = num_cells;
	stepDeadCells[0] = 0;
	stepNewCells[0] = 0;
	aliveList[0] = 0; //my_alive_cells lo del bucle 4.6
	maxFoodTemp[0] = 0;
	//cells3 = (Cell *)malloc( sizeof(Cell) * (size_t)num_cellsAux2 ); //cuidadoooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
	//cudaMalloc((void **) &cellsDevice3, sizeof(Cell) * (size_t)num_cellsAux2); //Cuidadoooooooooooooooooooooooooooooooo Esto era del tamaño de step new cells
	//cells3 = (Cell *)malloc( sizeof(Cell) * (size_t)step_new_cells );
	//cudaMemcpy( cellsDevice3, cells3, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice); //Cuidadoooooooooooooooooooooooooooooooo Esto era del tamaño de step new cells

	cudaMemcpy( p2Device, culture_cells, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );

	cudaMemcpy( statsDevice, stats, sizeof(Statistics) * 1 * 1 ,cudaMemcpyHostToDevice );


	cudaMemcpy( cellsDevice2, cells2, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

	cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

	cudaMemcpy( stepDeadCellsDevice, stepDeadCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	//cudaMemcpy( banderaDevice, bandera, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	cudaMemcpy( numeroCelulasDevice, numeroCelulas, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	/*
	int *food_to_share = (int *)malloc( sizeof(int) * num_cellsAux2 );
	if ( food_to_share == NULL ) {
		fprintf(stderr,"-- Error allocating food_to_share structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}*/

	//cudaMemcpy( foodShareDevice, food_to_share, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

	cudaMemcpy( numAliveDevice, numAlive, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	
	cudaMemcpy( stepDeadCellsDevice, stepDeadCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	//Cell *new_cells = (Cell *)malloc( sizeof(Cell) * num_cells );
	Cell *new_cells2 = (Cell *)malloc( sizeof(Cell) * num_cellsAux2 );
	if ( new_cells2 == NULL ) {
		fprintf(stderr,"-- Error allocating new cells structures for: %d cells\n", num_cells );
		exit( EXIT_FAILURE );
	}

	cudaMemcpy( newCellsDevice, new_cells2, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice);

	cudaMemcpy( stepNewCellsDevice, stepNewCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	cudaMemcpy( aDevice, aliveList, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	//cudaMemcpy( cellsDevice3, cells3, sizeof(Cell) * step_new_cells ,cudaMemcpyHostToDevice);
	
	cudaMemcpy( cDevice, maxFoodTemp, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

	double totalSuma2 = 0;
	double totalSuma3 = 0;
	double totalSuma5 = 0;
	double totalSuma6 = cp_Wtime();
	double totalSuma7 = 0;

for( iter=0; iter<max_iter && current_max_food <= max_food_int && num_cells_alive > 0; iter++ ) {
		
		//cudaMalloc((void **) &cellsDevice, sizeof(Cell) * (size_t)num_cells); //cells
		//int step_new_cells = 0;
		//int step_dead_cells = 0;
		

		if(iter!=0){
			cudaMemcpy( culture, pDevice, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		}

		double ttotal6 = cp_Wtime();
		
		/* 4.1. Spreading new food */
		// Across the whole culture
		int num_new_sources = (int)(rows * columns * food_density);

		//int* numNewSources = (int *)malloc( sizeof(int) * (size_t)num_new_sources*3);

		for (i=0; i<num_new_sources; i++) {
			int row = int_urand48( rows, food_random_seq );
			int col = int_urand48( columns, food_random_seq );
			int food = int_urand48( food_level * PRECISION, food_random_seq );
			accessMat( culture, row, col ) = accessMat( culture, row, col ) + food;
		}
		// In the special food spot
		if ( food_spot_active ) {
			num_new_sources = (int)(food_spot_size_rows * food_spot_size_cols * food_spot_density);
			for (i=0; i<num_new_sources; i++) {
				int row = food_spot_row + int_urand48( food_spot_size_rows, food_spot_random_seq );
				int col = food_spot_col + int_urand48( food_spot_size_cols, food_spot_random_seq );
				int food = int_urand48( food_spot_level * PRECISION, food_spot_random_seq );
				accessMat( culture, row, col ) = accessMat( culture, row, col ) + food;
			}
		}

		ttotal6 = cp_Wtime() - ttotal6;

		totalSuma7 += ttotal6;

		cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );

		

		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );

		//cudaMemcpy( p2Device, culture_cells, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );

		/* 4.2. Prepare ancillary data structures */
		/* 4.2.1. Clear ancillary structure of the culture to account alive cells in a position after movement */


		iniciarCultureCells<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(p2Device, rows*columns);

		//cudaMemcpy( culture_cells, p2Device, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );

		/*
		for( i=0; i<rows; i++ )
			for( j=0; j<columns; j++ ) 
				accessMat( culture_cells, i, j ) = 0;
		*/
		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
 		/* 4.2.2. Allocate ancillary structure to store the food level to be shared by cells in the same culture place */

 		/*
		int *food_to_share = (int *)malloc( sizeof(int) * num_cells );
		if ( food_to_share == NULL ) {
			fprintf(stderr,"-- Error allocating food_to_share structures for size: %d x %d \n", rows, columns );
			exit( EXIT_FAILURE );
		}
		*/

		//kernelito de inicialización
		double ttotal2 = cp_Wtime();
		//historyMaxAge[0] = sim_stat.history_max_age;
		inicializarDevices<<<1, 2, sizeof(int)*2>>>(stepDeadCellsDevice, stepNewCellsDevice, aDevice, cDevice);

		ttotal2 = cp_Wtime() - ttotal2;

		totalSuma2 += ttotal2;
		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		//cudaMemcpy( p2Device, culture_cells, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		//cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

		//printf
		//cudaMemcpy( foodShareDevice, food_to_share, sizeof(int) * num_cells ,cudaMemcpyHostToDevice );
		//cudaMemcpy( numAliveDevice, numAlive, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		//cudaMemcpy( stepDeadCellsDevice, stepDeadCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		//cudaMemcpy( historyMaxAgeDevice, historyMaxAge, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		
		double ttotal3 = cp_Wtime();
		cudaMemcpy( bandera, numeroCelulasDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );



		movimientoCelulas<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, rows, columns, numAliveDevice, stepDeadCellsDevice, statsDevice, numeroCelulasDevice, pDevice, p2Device/*, foodShareDevice*/);

		ttotal3 = cp_Wtime() - ttotal3;

		totalSuma3 += ttotal3;
		//cudaMemcpy( stats, statsDevice, sizeof(Statistics) * 1 * 1 ,cudaMemcpyHostToDevice );

		//cudaMemcpy( numAlive, numAliveDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( stepDeadCells, stepDeadCellsDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( historyMaxAge, historyMaxAgeDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( cells, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );

		/*
		for(i=0; i<num_cells;i++){
			if(cells[i].age>30){
				printf("hola  %i    en la iteracion  %i    storage     %i\n", cells[i].age, iter, cells[i].storage);
			}
		}*/

		//movimientoCelulas2<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, pDevice, p2Device, foodShareDevice, numeroCelulasDevice, columns);

		//cudaMemcpy( cells, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( culture, pDevice, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( culture_cells, p2Device, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( food_to_share, foodShareDevice, sizeof(int) * num_cells ,cudaMemcpyDeviceToHost );


		/*	
		int s = 0;
		int a = 0;
		for(i=0; i<num_cells; i++){
			s += food_to_share[i];
			a += accessMat( culture_cells, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION );
		}
		printf("suma del food to share %i y la suma de posiciones %i en la iteracion %i\n", s, a, iter);*/
		
		//sim_stat.history_max_age = historyMaxAge[0];
		//num_cells_alive = numAlive[0];
		//step_dead_cells = stepDeadCells[0];

		/*
		for (i=0; i<num_cells; i++) {
			if(cells[i].alive){
				accessMat( culture_cells, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION ) += 1;
				food_to_share[i] = accessMat( culture, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION );
			}
		}*/
		/* 4.3. Cell movements */
		/*
		for (i=0; i<num_cells; i++) {
			if ( cells[i].alive ) {
				cells[i].age ++;
				// Statistics: Max age of a cell in the simulation history
				if ( cells[i].age > sim_stat.history_max_age ) sim_stat.history_max_age = cells[i].age;

				/* 4.3.1. Check if the cell has the needed energy to move or keep alive */
				/*if ( cells[i].storage < ENERGY_NEEDED_TO_LIVE ) {
					// Cell has died
					cells[i].alive = false;
					num_cells_alive --;
					step_dead_cells ++;
					continue;
				}
				if ( cells[i].storage < ENERGY_NEEDED_TO_MOVE ) {
					// Almost dying cell, it cannot move, only if enough food is dropped here it will survive
					cells[i].storage -= ENERGY_SPENT_TO_LIVE;
				}
				else {
					// Consume energy to move
					cells[i].storage -= ENERGY_SPENT_TO_MOVE;
						
					/* 4.3.2. Choose movement direction */
					/*int prob = int_urand48( PRECISION, cells[i].random_seq );
					if ( prob < cells[i].choose_mov[0] ) {
						// Turn left (90 degrees)
						int tmp = cells[i].mov_col;
						cells[i].mov_col = cells[i].mov_row;
						cells[i].mov_row = -tmp;
					}
					else if ( prob >= cells[i].choose_mov[0] + cells[i].choose_mov[1] ) {
						// Turn right (90 degrees)
						int tmp = cells[i].mov_row;
						cells[i].mov_row = cells[i].mov_col;
						cells[i].mov_col = -tmp;
					}
					// else do not change the direction
					
					/* 4.3.3. Update position moving in the choosen direction*/
					/*cells[i].pos_row += cells[i].mov_row;
					cells[i].pos_col += cells[i].mov_col;
					// Periodic arena: Left/Rigth edges are connected, Top/Bottom edges are connected
					if ( cells[i].pos_row < 0 ) cells[i].pos_row += rows * PRECISION;
					if ( cells[i].pos_row >= rows * PRECISION) cells[i].pos_row -= rows * PRECISION;
					if ( cells[i].pos_col < 0 ) cells[i].pos_col += columns * PRECISION;
					if ( cells[i].pos_col >= columns * PRECISION) cells[i].pos_col -= columns * PRECISION;
				}

				/* 4.3.4. Annotate that there is one more cell in this culture position */
				//accessMat( culture_cells, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION ) += 1;
				/* 4.3.5. Annotate the amount of food to be shared in this culture position */
				//food_to_share[i] = accessMat( culture, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION );
			//}
		//} // End cell movements
		
		/* 4.4. Cell actions */
		// Space for the list of new cells (maximum number of new cells is num_cells)

		//pal kernelito de inicializacion buffer

		
		//historyTotalCells[0] = sim_stat.history_total_cells;

		//cudaMemcpy( newCellsDevice, new_cells2, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice);
		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		//cudaMemcpy( p2Device, culture_cells, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		//cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );
		//cudaMemcpy( foodShareDevice, food_to_share, sizeof(int) * num_cells ,cudaMemcpyHostToDevice );
		//cudaMemcpy( numAliveDevice, numAlive, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		//cudaMemcpy( stepNewCellsDevice, stepNewCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		//cudaMemcpy( historyTotalCellsDevice, historyTotalCells, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );

		double ttotal4 = cp_Wtime();

		nacimientoCelulas<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, p2Device, pDevice/*, foodShareDevice*/, newCellsDevice, numAliveDevice,
			stepNewCellsDevice, banderaDevice, numeroCelulasDevice, columns, statsDevice);

		ttotal4 = cp_Wtime() - ttotal4;


		//cudaMemcpy( bandera, banderaDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		//printf("bandera valor %i\n", bandera[0]);

		//actualizarStats<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(banderaDevice,statsDevice);
		//cudaMemcpy( cells, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( new_cells2, newCellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );	
		//cudaMemcpy( culture, pDevice, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( culture_cells, p2Device, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( numAlive, numAliveDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( stepNewCells, stepNewCellsDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		
		/*
		for(i=0; i<num_cells; i++){
			new_cells[i] = new_cells2[i];
		}
		*/
		//sim_stat.history_total_cells = historyTotalCells[0];
		//num_cells_alive = numAlive[0];
		//step_new_cells = stepNewCells[0];
		
		//cudaMemcpy( bandera, numeroCelulasDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		//cudaMalloc((void **) &newCellsDeviceTemp, sizeof(Cell) * (size_t)bandera[0]); //cells entrada
		
		//ajustar<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(newCellsDevice, newCellsDeviceTemp, numeroCelulasDevice);

		//cudaFree(newCellsDevice);

		//cudaMalloc((void **) &newCellsDevice, sizeof(Cell) * (size_t)bandera[0]);

		//ajustar<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(newCellsDeviceTemp, newCellsDevice, numeroCelulasDevice);

		//cudaFree(newCellsDeviceTemp);

		/*
		for (i=0; i<num_cells; i++) {
			if ( cells[i].alive ) {
				/* 4.4.1. Food harvesting */
			/*
				int food = food_to_share[i];
				int count = accessMat( culture_cells, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION );
				int my_food = food / count;
				cells[i].storage += my_food;

				/* 4.4.2. Split cell if the conditions are met: Enough maturity and energy */
		/*
				if ( cells[i].age > 30 && cells[i].storage > ENERGY_NEEDED_TO_SPLIT ) {
					// Split: Create new cell
					num_cells_alive ++;
					sim_stat.history_total_cells ++;
					step_new_cells ++;

					// New cell is a copy of parent cell
					new_cells[ step_new_cells-1 ] = cells[i];

					// Split energy stored and update age in both cells
					cells[i].storage /= 2;
					new_cells[ step_new_cells-1 ].storage /= 2;
					cells[i].age = 1;
					new_cells[ step_new_cells-1 ].age = 1;

					// Random seed for the new cell, obtained using the parent random sequence
					new_cells[ step_new_cells-1 ].random_seq[0] = (unsigned short)glibc_nrand48( cells[i].random_seq );
					new_cells[ step_new_cells-1 ].random_seq[1] = (unsigned short)glibc_nrand48( cells[i].random_seq );
					new_cells[ step_new_cells-1 ].random_seq[2] = (unsigned short)glibc_nrand48( cells[i].random_seq );

					// Both cells start in random directions
					cell_new_direction( &cells[i] );
					cell_new_direction( &new_cells[ step_new_cells-1 ] );
				
					// Mutations of the movement genes in both cells
					cell_mutation( &cells[i] );
					cell_mutation( &new_cells[ step_new_cells-1 ] );
				}
			}
		} // End cell actions

		/*
		if(iter == 1){
			for(i=0; i<rows*columns; i++){
				printf("que esta pasando? %i\n", culture[i]);
			}
		}*/
		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		//cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );
		
		comerComida<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(pDevice, cellsDevice, numeroCelulasDevice, rows, columns);

		//cudaMemcpy( cells, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );
		//cudaMemcpy( culture, pDevice, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );

		/*if(iter == 1){
			for(i=0; i<rows*columns; i++){
				printf("esto esta pasando %i\n", culture[i]);
			}
		}*/
		/* 4.5. Clean ancillary data structures */
		/* 4.5.1. Clean the food consumed by the cells in the culture data structure */
		/*
		for (i=0; i<num_cells; i++) {
			if ( cells[i].alive ) {
				accessMat( culture, cells[i].pos_row / PRECISION, cells[i].pos_col / PRECISION ) = 0;
			}
		}*/
		/* 4.5.2. Free the ancillary data structure to store the food to be shared */
		//free( food_to_share );

		/* 4.6. Clean dead cells from the original list */
		// 4.6.1. Move alive cells to the left to substitute dead cells

		//pal kernelito de reset

		//int alive_in_main_list = 0;


		//cudaMemcpy( aDevice, aliveList, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		//cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

		/*
		int p = 0;
		for(int k = 0; k<num_cells; k++){
			if(cells[k].alive){
				p ++ ;
			}
		}
		if(iter==10){
			printf("numero de celulas vivas %i   en la iteracion %i  numero de celulas  %i\n", p, iter, num_cells);
		}*/
		limpiarCelulas<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, numeroCelulasDevice, aDevice, cellsDevice2);
		
		cudaMemcpy( bandera, aDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		numeroBloq2 = bandera[0]/tamanoBloq;
		if (bandera[0]%tamanoBloq!=0) 
    	{
        	numeroBloq2++;
    	}

		limpiarCelulas2<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice2, aDevice, cellsDevice, numeroCelulasDevice);

		//cudaFree(cellsDevice2);

		//cudaMemcpy( bandera, numeroCelulasDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		//cudaMalloc((void **) &cellsDevice2, sizeof(Cell) * (size_t)num_cellsAux2); //cells entrada

		/*
		if(num_cellsAux<bandera[0]){

			num_cellsAux = bandera[0];

			cudaMalloc((void **) &newCellsDeviceTemp2, sizeof(Cell) * (size_t)bandera[0]); //cells entrada
			
			ajustar<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, newCellsDeviceTemp2, numeroCelulasDevice);

			cudaFree(cellsDevice);

			cudaMalloc((void **) &cellsDevice, sizeof(Cell) * (size_t)bandera[0]);

			ajustar<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(newCellsDeviceTemp2, cellsDevice, numeroCelulasDevice);

			cudaFree(newCellsDeviceTemp2);
		}*/
		
		//si seguimos la explicacion de abajo, aqui habria que combinar cellsDevice2 y cellsDevice con limpiarCelulas2
		//cudaMemcpy( cells2, cellsDevice2, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );

		//cudaMemcpy( aliveList, aDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		/*
		for (i = 0; i<num_cells; i++){
			cells[i] = cells2[i];
		}*/

		/*
		Cell* cells5 = (Cell *)malloc( sizeof(Cell) * (size_t)num_cells );
		cudaMemcpy( cells5, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );
		for(i = 0; i < num_cells; i++){
			cells[i] = cells5[i];
		}
		*/

		/*
		if(iter == 10){
			printf("posicion%i\n", free_position);
		}*/

		//alive_in_main_list = aliveList[0];

		/*p = 0;
		for(int k = 0; k<num_cells; k++){
			if(cells[k].alive){
				p ++ ;
			}
		}
		if(iter==10){
			printf("numero de celulas vivas %i   en la iteracion %i  numero de celulas  %i alive_in_main_list  %i\n", p, iter, num_cells, alive_in_main_list);
		}*/
		//printf("hay muchas vivas %i\n", alive_in_main_list);
		/*
		for( i=0; i<num_cells; i++ ) {
			if ( cells[i].alive ) {
				alive_in_main_list ++;
				if ( free_position != i ) {
					cells[free_position] = cells[i];
				}
				free_position ++;
			}
		}*/
		// 4.6.2. Reduce the storage space of the list to the current number of cells
		//num_cells = alive_in_main_list;
		/*
		if(iter==10){
			printf("numero de celulas vivas finales %i\n", num_cells);
		}*/
		//cells = (Cell *)realloc( cells, sizeof(Cell) * num_cells );
		/*
		p = 0;
		for(int k = 0; k<num_cells; k++){
			if(cells[k].alive){
				p ++ ;
			}
		}
		if(iter==10){
			printf("ssssssssssnumero de celulas vivas %i   en la iteracion %i  numero de celulas  %i alive_in_main_list  %i\n", p, iter, num_cells, alive_in_main_list);
		}*/
		//if ( step_new_cells > 0 ) {
			//cells = (Cell *)realloc( cells, sizeof(Cell) * ( num_cells + step_new_cells ) );
		//}
		
		/*
		for(i = 0; i < 5; i++){
			printf("edad antes %i\n", new_cells[i].age);
		}*/

		/*
		for(i = num_cells; i < num_cells + 5; i++){
			printf("edad anntes segunda %i  posicion   %i\n", cells[i].age, i);
		}*/

		/*
		cudaMalloc((void **) &cellsDevice3, sizeof(Cell) * (size_t)step_new_cells); //cells salida 2
		cells3 = (Cell *)malloc( sizeof(Cell) * (size_t)step_new_cells );
		cudaMemcpy( cellsDevice3, cells3, sizeof(Cell) * step_new_cells ,cudaMemcpyHostToDevice);
		*/
		/* 4.7. Join cell lists: Old and new cells list */

		//cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );
		//No hace falta pasar cellsDevice, por que la hemos fusionado en la gpu en el anterior paso con limpiarCelulas2 (es entre comillas el obteivo final de la practica)
		//Trabajar con todo en la cpu, solo con devices, no existira cells ni culture_cells, solo culture
		//cudaMemcpy( newCellsDevice, new_cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice);

		//HACER ESTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

		//cudaMalloc((void **) &cellsDevice3, sizeof(Cell) * (size_t)num_cells); //Cuidadoooooooooooooooooooooooooooooooo Esto era del tamaño de step new cells
		
		//cudaMemcpy( cellsDevice3, cells3, sizeof(Cell) * step_new_cells ,cudaMemcpyHostToDevice);

		//declararMemoria( )

		//HACER ESTOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

		anyadirCelulas<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(stepNewCellsDevice, newCellsDevice, cellsDevice2);

		cudaMemcpy( bandera2, stepNewCellsDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		bandera[0]+=bandera2[0];
		numeroBloq2 = bandera[0]/tamanoBloq;
		if (bandera[0]%tamanoBloq!=0) 
    	{
        	numeroBloq2++;
    	}

		//cudaMemcpy( cells3, cellsDevice3, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( cellsDevice3, cells3, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

		//cudaMemcpy( bandera2, stepNewCellsDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		/*
		if(num_cellsAux < bandera[0] + bandera2[0]){
			num_cellsAux = bandera[0] + bandera2[0];

			cudaMalloc((void **) &newCellsDeviceTemp2, sizeof(Cell) * (size_t)bandera[0] + bandera2[0]); //cells entrada
			
			ajustar2<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, newCellsDeviceTemp2, numeroCelulasDevice, stepNewCellsDevice);

			cudaFree(cellsDevice);

			cudaMalloc((void **) &cellsDevice, sizeof(Cell) * (size_t)bandera[0] + bandera2[0]);

			ajustar2<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(newCellsDeviceTemp2, cellsDevice, numeroCelulasDevice, stepNewCellsDevice);

			cudaFree(newCellsDeviceTemp2);
		}*/


		//Importante***************************************************************************************************************************************************************************
		//Para cuando todo este paralelizado no hara falta traerse la matriz cells, combinamos cellsDevice con cellsDevice3 y ya 
		//Habra que combinar los cellsDevices que se vayan obteniendo de salida con el cellDevice estandar que seria el array cells de la gpu
		//Por ejemplo, en anyadirCelulas se deja en cellsDevices3 las celulas que se van a anyadir, habria que fusionarlo con cellsDevice que es lo que se hace aqui debajo, mantenidendo
		//asi en cellsDevice siempre el array de celulas bueno
		anyadirCelulas2<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice2, stepNewCellsDevice, numeroCelulasDevice, cellsDevice); //para unir el cellsDevice3 al cellsDevice
		//cudaCheckLast();

		actualizarNumCells<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(numeroCelulasDevice, stepNewCellsDevice);
		/*
		cudaMemcpy( cells3, cellsDevice3, sizeof(Cell) * step_new_cells ,cudaMemcpyDeviceToHost );
		for(i = 0; i < step_new_cells; i++){
			cells[num_cells + i] = cells3[i];
		}*/
		/*
		for(i = 0; i < step_new_cells; i++){
			cellsDevice[num_cells + i] = cellsDevice3[i];
		}*/
		//cudaMemcpy( cells, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );

	
		//para probar lo explicado arriba
		/*
		num_cells += step_new_cells;
		Cell* cells4 = (Cell *)malloc( sizeof(Cell) * (size_t)num_cells );
		cudaMemcpy( cells4, cellsDevice, sizeof(Cell) * num_cells ,cudaMemcpyDeviceToHost );
		for(i = 0; i < num_cells; i++){
			cells[i] = cells4[i];
		}
		*/
		/*
		for(i = num_cells - 5; i < num_cells; i++){
			printf("edad despues %i  posicion   %i\n", cells[i].age, i);
		}*/

		/*
		if ( step_new_cells > 0 ) {
			cells = (Cell *)realloc( cells, sizeof(Cell) * ( num_cells + step_new_cells ) );
			for (j=0; j<step_new_cells; j++)
				cells[ num_cells + j ] = new_cells[ j ];
			num_cells += step_new_cells;
		}*/
		//free( new_cells );
		/*
		current_max_food = 0;

		maxFoodTemp[0] = current_max_food;
		cudaMemcpy( cDevice, maxFoodTemp, sizeof(int) * 1 * 1 ,cudaMemcpyHostToDevice );
		*/
		//cudaMemcpy( pDevice, culture, sizeof(int) * rows * columns ,cudaMemcpyHostToDevice );
		
		double ttotal5 = cp_Wtime();

		reductionMax<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(pDevice, rows*columns, cDevice);

		ttotal5 = cp_Wtime() - ttotal5;

		totalSuma5 += ttotal5;

		//cudaMemcpy( culture, pDevice, sizeof(int) * rows * columns ,cudaMemcpyDeviceToHost );
		//cudaMemcpy( maxFoodTemp, cDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );

		//current_max_food = maxFoodTemp[0]; //La primera variable resultado

		/* 4.8. Decrease non-harvested food */
		/*
		for( i=0; i<rows; i++ )
			for( j=0; j<columns; j++ ) {
				accessMat( culture, i, j ) -= accessMat( culture, i, j ) / 20;
				if ( accessMat( culture, i, j ) > current_max_food ) 
					current_max_food = accessMat( culture, i, j );
			}
		*/
		/* 4.9. Statistics */
		// Statistics: Max food


		final<<<1, 2, sizeof(int)*2>>>(cDevice, stepNewCellsDevice, stepDeadCellsDevice, numAliveDevice, statsDevice);

		//cudaMemcpy( numAlive, numAliveDevice, sizeof(int) * 1 * 1 ,cudaMemcpyDeviceToHost );
		cudaMemcpy( stats, statsDevice, sizeof(Statistics) * 1 * 1 ,cudaMemcpyDeviceToHost );

		//stats[0] = sim_stat;
		current_max_food = stats[0].history_max_food;
		num_cells_alive = bandera[0];
		//printf("step_dead_cells  %i  step_new_cells   %i    current_max_food    %i     num_cells_alive     %i\n",step_dead_cells, step_new_cells, current_max_food, num_cells_alive);

		/*
		if ( current_max_food > sim_stat.history_max_food ) sim_stat.history_max_food = current_max_food;
		// Statistics: Max new cells per step
		if ( step_new_cells > sim_stat.history_max_new_cells ) sim_stat.history_max_new_cells = step_new_cells;
		// Statistics: Accumulated dead and Max dead cells per step
		sim_stat.history_dead_cells += step_dead_cells;
		if ( step_dead_cells > sim_stat.history_max_dead_cells ) sim_stat.history_max_dead_cells = step_dead_cells;
		// Statistics: Max alive cells per step
		if ( num_cells_alive > sim_stat.history_max_alive_cells ) sim_stat.history_max_alive_cells = num_cells_alive;
		//numAlive[0] = num_cells_alive;
		*/

		

		//udaMemcpy( statsDevice, stats, sizeof(Statistics) * 1 * 1 ,cudaMemcpyHostToDevice );

		//stepDeadCells[0] = 0;
		//stepNewCells[0] = 0;
		//maxFoodTemp[0] = 0;

#ifdef DEBUG
		/* 4.10. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, culture, num_cells, cells, num_cells_alive, sim_stat );
#endif // DEBUG
	}

	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	// CUDA stop
	cudaDeviceSynchronize();

	/* 5. Stop global time */
	ttotal = cp_Wtime() - ttotal;

	totalSuma6 = cp_Wtime() - totalSuma6;

#ifdef DEBUG
	printf("List of cells at the end of the simulation: %d\n\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("Cell %d, Alive: %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i,
				cells[i].alive,
				(float)cells[i].pos_row / PRECISION, 
				(float)cells[i].pos_col / PRECISION, 
				(float)cells[i].mov_row / PRECISION, 
				(float)cells[i].mov_col / PRECISION, 
				(float)cells[i].choose_mov[0] / PRECISION, 
				(float)cells[i].choose_mov[1] / PRECISION, 
				(float)cells[i].choose_mov[2] / PRECISION, 
				(float)cells[i].storage / PRECISION,
				cells[i].age );
	}
#endif // DEBUG

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	//printf("Tiempo del bucle de inicializacion: %lf\n", totalSuma2);

	//printf("Tiempo del bucle de movimiento: %lf\n", totalSuma3);

	//printf("Tiempo del bucle de nacimiento: %lf\n", totalSuma4);

	//printf("Tiempo del bucle de 4.8: %lf\n", totalSuma5);

	//printf("Tiempo del bucle de 4.1: %lf\n", totalSuma7);

	//printf("Secuencia principal: %lf\n", totalSuma6);

	/* 6.2. Results: Number of iterations and other statistics */
	cudaMemcpy( stats, statsDevice, sizeof(Statistics) * 1 * 1 ,cudaMemcpyDeviceToHost );
	sim_stat = stats[0];
	printf("Result: %d, ", iter);
	printf("%d, %d, %d, %d, %d, %d, %d, %f\n", 
		num_cells_alive, 
		sim_stat.history_total_cells, 
		sim_stat.history_dead_cells, 
		sim_stat.history_max_alive_cells, 
		sim_stat.history_max_new_cells, 
		sim_stat.history_max_dead_cells, 
		sim_stat.history_max_age,
		(float)sim_stat.history_max_food / PRECISION
	);

	/* 7. Free resources */	
	free( culture );
	free( culture_cells );
	free( cells );

	/* 8. End */
	return 0;
}