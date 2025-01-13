#include "IterationSpacesSfc.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::IterationSpacesSfc::init( std::vector< dim_t >   const * i_loop_dim_type,
                                                   std::vector< exec_t >  const * i_loop_exec_type,
                                                   std::vector< int64_t > const * i_loop_sizes,
                                                   std::vector< int64_t > const * i_loop_strides_left,
                                                   std::vector< int64_t > const * i_loop_strides_right,
                                                   std::vector< int64_t > const * i_loop_strides_out,
                                                   int64_t                        i_num_threads){

  m_loop_dim_type        = i_loop_dim_type;
  m_loop_exec_type       = i_loop_exec_type;
  m_loop_sizes           = i_loop_sizes;

  m_loop_strides.resize(MAX_NUM_INPUTS);
  m_loop_strides[LEFT] = i_loop_strides_left;
  m_loop_strides[RIGHT] = i_loop_strides_right;
  m_loop_strides[OUT] = i_loop_strides_out;

  m_num_threads = i_num_threads;
}

einsum_ir::err_t einsum_ir::backend::IterationSpacesSfc::compile(){
 err_t l_err = err_t::UNDEFINED_ERROR;

  //calculate number of generated tasks
  m_num_tasks = 1;
  m_num_parallel_loops = 0;
  range_t l_parallel_loops; 
  for( size_t l_id = 0; l_id < m_loop_dim_type->size(); l_id++ ){
    if( m_loop_exec_type->at(l_id) == einsum_ir::OMP ||
        m_loop_exec_type->at(l_id) == einsum_ir::SFC    ){
      if( !m_num_parallel_loops ){
        l_parallel_loops.begin = l_id;
      }
      m_num_tasks *= m_loop_sizes->at(l_id);
      m_num_parallel_loops += 1;
    }
  }
  l_parallel_loops.end = l_parallel_loops.begin + m_num_parallel_loops;

  //assigns parallel dimensions to three types omp, sfc_n, sfc_m
  //restrictions:
  //  all parallel dims must be consecutive
  //  first omp dims can be of type m,n or c
  //  second sfc dims of type n
  //  third sfc dims of type m
  // example: 
  //  dim_t : ...  c1,  m1,  n1,  n2,  m2,  m3, ...
  //  exec_t: ... omp, omp, omp, sfc, sfc, sfc, ...
  int64_t l_omp_tasks = 1;
  int64_t l_sfc_tasks_n = 1;
  int64_t l_sfc_tasks_m = 1;
  int64_t l_last_found_type = 0;
  range_t l_omp_loops;
  range_t l_sfc_loops_n;
  range_t l_sfc_loops_m;
  for( int64_t l_id = l_parallel_loops.begin; l_id < l_parallel_loops.end ; l_id++ ){
    if( m_loop_exec_type->at(l_id) == einsum_ir::OMP &&
        l_last_found_type      <= 1 ){
      l_omp_tasks *= m_loop_sizes->at(l_id);
      if( l_last_found_type == 0 ){
        l_omp_loops.begin = l_id;
      }
      l_omp_loops.end = l_id + 1;
      l_last_found_type = 1;
    }
    else if( m_loop_exec_type->at(l_id) == einsum_ir::SFC &&
             m_loop_dim_type->at(l_id)  == einsum_ir::N   &&
             l_last_found_type <= 2 ){
      l_sfc_tasks_n *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 1 ){
        l_sfc_loops_n.begin = l_id;
      }
      l_sfc_loops_n.end = l_id + 1;
      l_last_found_type = 2;
    }
    else if( m_loop_exec_type->at(l_id) == einsum_ir::SFC &&
             m_loop_dim_type->at(l_id)  == einsum_ir::M ){
      l_sfc_tasks_m *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 2 ){
        l_sfc_loops_m.begin = l_id;
      }
      l_sfc_loops_m.end = l_id + 1;
      l_last_found_type = 3;
    }
    else{
      return err_t::COMPILATION_FAILED;
    }
  }
 
  //convert strides to offsets
  m_movement_offsets.resize(MAX_NUM_INPUTS);
  for(int64_t l_input = 0; l_input < MAX_NUM_INPUTS; l_input++){
    convertStridesToOffsets( l_omp_loops,
                             l_sfc_loops_m,
                             l_sfc_loops_n,
                             l_sfc_tasks_m > l_sfc_tasks_n,
                             *m_loop_strides[l_input],
                             m_movement_offsets[l_input] );
  } 
  
  //allocate memory for iteration space
  m_dim_movements.resize(   m_num_threads );
  m_initial_offsets.resize( m_num_threads );
  std::vector< range_t > l_thread_work_space;
  l_thread_work_space.resize( m_num_threads );
  int64_t l_tasks_per_thread = m_num_tasks / m_num_threads + (m_num_tasks % m_num_threads != 0);
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_id * l_tasks_per_thread;
    int64_t l_end   = l_begin     + l_tasks_per_thread;
    l_begin = l_begin < m_num_tasks ? l_begin : m_num_tasks;
    l_end   = l_end   < m_num_tasks ? l_end   : m_num_tasks;

    l_thread_work_space[l_thread_id].begin = l_begin;
    l_thread_work_space[l_thread_id].end   = l_end;

    m_dim_movements[l_thread_id].resize( l_end - l_begin );
    m_initial_offsets[l_thread_id].resize( MAX_NUM_INPUTS );
  }

//create 1D Map of task
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_work_space[l_thread_id].begin;
    int64_t l_end   = l_thread_work_space[l_thread_id].end;

    int l_id_sfc_m_old, l_id_sfc_n_old, l_id_omp_old;
    gilbert_d3xy(&l_id_sfc_m_old, &l_id_sfc_n_old, &l_id_omp_old, l_begin, l_sfc_tasks_m, l_sfc_tasks_n);

    //calculate initial thread offsets
    for(int64_t l_input = 0; l_input < MAX_NUM_INPUTS; l_input++){
      int64_t l_offset = calculateInitialOffsets( l_omp_loops,
                                                  l_sfc_loops_m,
                                                  l_sfc_loops_n,
                                                  l_id_omp_old,
                                                  l_id_sfc_m_old,
                                                  l_id_sfc_n_old,
                                                  *m_loop_strides[l_input] );
      m_initial_offsets[l_thread_id][l_input] = l_offset;
    }

    //calculate movements
    for( int64_t l_id = l_begin; l_id < l_end; l_id++ ){
      int l_id_sfc_m_new, l_id_sfc_n_new, l_id_omp_new;
      gilbert_d3xy(&l_id_sfc_m_new, &l_id_sfc_n_new, &l_id_omp_new, l_id+1, l_sfc_tasks_m, l_sfc_tasks_n);

      if( l_id_omp_new != l_id_omp_old ){
        uint8_t l_move = getMaxDimJump(l_omp_loops, l_id_omp_new, l_id_omp_old, l_omp_loops.begin);
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move; 
      }
      else if( l_id_sfc_m_new != l_id_sfc_m_old && l_id_sfc_n_new != l_id_sfc_n_old ){
        //in case of a jump in sfc treat as moving in a new direction
        uint8_t l_move_m = getMaxDimJump(l_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old, l_omp_loops.begin);
        uint8_t l_move_n = getMaxDimJump(l_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old, l_omp_loops.begin);
        int64_t l_direction_m = l_move_m >> 7 ? -1 : 1;
        int64_t l_direction_n = l_move_n >> 7 ? -1 : 1;
        l_move_m = l_direction_m == 1 ? l_move_m : 256-l_move_m;
        l_move_n = l_direction_n == 1 ? l_move_n : 256-l_move_n;

        m_dim_movements[l_thread_id][l_id-l_begin] = m_num_parallel_loops + 1;
        for(int64_t l_input = 0; l_input < MAX_NUM_INPUTS; l_input++){
          int64_t l_offset = 0;
          l_offset += l_direction_m * m_movement_offsets[l_input][l_move_m];
          l_offset += l_direction_n * m_movement_offsets[l_input][l_move_n];
          m_movement_offsets[l_input][m_num_parallel_loops + 1] = l_offset;
        }
      }
      else if( l_id_sfc_m_new != l_id_sfc_m_old ){
        uint8_t l_move = getMaxDimJump(l_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old, l_omp_loops.begin);
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;

      }
      else if( l_id_sfc_n_new != l_id_sfc_n_old ){
        uint8_t l_move = getMaxDimJump(l_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old, l_omp_loops.begin);
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }
      l_id_sfc_m_old = l_id_sfc_m_new;
      l_id_sfc_n_old = l_id_sfc_n_new;
      l_id_omp_old = l_id_omp_new;
    }
  }

  return err_t::SUCCESS;
}

int64_t einsum_ir::backend::IterationSpacesSfc::calculateInitialOffsets( range_t i_omp_loops,
                                                                         range_t i_sfc_loops_m,
                                                                         range_t i_sfc_loops_n,
                                                                         int64_t i_id_omp,
                                                                         int64_t i_id_sfc_m,
                                                                         int64_t i_id_sfc_n,
                                                                         std::vector< int64_t > const & i_strides ) {

  int64_t l_offset = 0;
  for (int64_t l_id = i_sfc_loops_m.end - 1; l_id >= i_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_m % l_size) * l_stride;
    i_id_sfc_m /= l_size;
  }
  for (int64_t l_id = i_sfc_loops_n.end - 1; l_id >= i_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_n % l_size) * l_stride;
    i_id_sfc_n /= l_size;
  }
  for (int64_t l_id = i_omp_loops.end - 1; l_id >= i_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_omp % l_size) * l_stride;
    i_id_omp /= l_size;
  }

  return l_offset;
}




void einsum_ir::backend::IterationSpacesSfc::convertStridesToOffsets( range_t i_omp_loops,
                                                                      range_t i_sfc_loops_m,
                                                                      range_t i_sfc_loops_n,
                                                                      bool    i_sfc_m_large,
                                                                      std::vector< int64_t > const & i_strides,
                                                                      std::vector< int64_t >       & io_offsets ) {
  
  // id 0 is always set to 0
  // last id can be used for diagonal jumps
  io_offsets.resize( m_num_parallel_loops + 2); 
  io_offsets[0] = 0;
  int64_t l_id_offset = m_num_parallel_loops;
  
  int64_t l_all_offsets_sfc_m = 0;
  for (int64_t l_id = i_sfc_loops_m.end - 1; l_id >= i_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id_offset ] = l_stride - l_all_offsets_sfc_m;
    l_all_offsets_sfc_m += (l_size - 1) * l_stride;
    l_id_offset--;
  }

  int64_t l_all_offsets_sfc_n = 0;
  for (int64_t l_id = i_sfc_loops_n.end - 1; l_id >= i_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id_offset ] = l_stride - l_all_offsets_sfc_n;
    l_all_offsets_sfc_n += (l_size - 1) * l_stride;
    l_id_offset--;
  }

  int64_t l_all_offsets_omp = i_sfc_m_large ? l_all_offsets_sfc_m : l_all_offsets_sfc_n;
  for (int64_t l_id = i_omp_loops.end - 1; l_id >= i_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id_offset ] = l_stride - l_all_offsets_omp;
    l_all_offsets_omp += (l_size - 1) * l_stride;
    l_id_offset--;
  }
}

uint8_t einsum_ir::backend::IterationSpacesSfc::getMaxDimJump( range_t i_dim_loops,
                                                               int64_t i_id_new,
                                                               int64_t i_id_old,
                                                               int64_t i_offset ){

  int64_t l_dif = i_id_new - i_id_old;
  int64_t l_max_id = i_id_new > i_id_old ? i_id_new : i_id_old;
  for( int64_t l_di = i_dim_loops.end-1; l_di >= i_dim_loops.begin; l_di-- ){
    int64_t l_size = m_loop_sizes->at(l_di);
    if(l_max_id % l_size != 0){
      return (l_di - i_offset + 1) * l_dif;
    }
    else{
      l_max_id /= l_size;
    }
  }

  return 0;
}

int64_t einsum_ir::backend::IterationSpacesSfc::getNumTasks( int64_t i_thread_id ){
  return m_dim_movements[i_thread_id].size();
}

void einsum_ir::backend::IterationSpacesSfc::addMovementOffsets( int64_t    i_thread_id, 
                                                                 int64_t    i_task_id,
                                                                 char    ** io_offset_left,
                                                                 char    ** io_offset_right,
                                                                 char    ** io_offset_out){
  uint8_t l_move =  m_dim_movements[i_thread_id][i_task_id];
  int64_t l_direction = l_move >> 7 ? -1 : 1;
  l_move = l_direction == 1 ? l_move : 256-l_move;

  *io_offset_left  += l_direction * m_movement_offsets[LEFT ][l_move];
  *io_offset_right += l_direction * m_movement_offsets[RIGHT][l_move];
  *io_offset_out   += l_direction * m_movement_offsets[OUT  ][l_move];
}

void einsum_ir::backend::IterationSpacesSfc::addInitialOffsets( int64_t    i_thread_id,
                                                                char    ** io_offset_left,
                                                                char    ** io_offset_right,
                                                                char    ** io_offset_out) {
  *io_offset_left  += m_initial_offsets[i_thread_id][LEFT ];
  *io_offset_right += m_initial_offsets[i_thread_id][RIGHT];
  *io_offset_out   += m_initial_offsets[i_thread_id][OUT  ];
}


#define SIGN(A) (0 < (A) ? (1) : ( 0 == (A) ? (0) : (-1)))

int gilbert_d2xy_r(int dst_idx, int cur_idx,
                       int *xres, int *yres,
                       int ax,int ay,
                       int bx,int by ) {
  int nxt_idx;
  int w, h, x, y,
      dax, day,
      dbx, dby,
      di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = abs(ax + ay);
  h = abs(bx + by);

  x = *xres;
  y = *yres;

  // unit major direction
  dax = SIGN(ax);
  day = SIGN(ay);

  // unit orthogonal direction
  dbx = SIGN(bx);
  dby = SIGN(by);

  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax*di;
    *yres = y + day*di;
    return 0;
  }

  if (w == 1) {
    *xres = x + dbx*di;
    *yres = y + dby*di;
    return 0;
  }

  // floor function
  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;

  w2 = abs(ax2 + ay2);
  h2 = abs(bx2 + by2);

  if ((2*w) > (3*h)) {
    if ((w2 & 1) && (w > 2)) {
      // prefer even steps
      ax2 += dax;
      ay2 += day;
    }

    // long case: split in two parts only
    nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return gilbert_d2xy_r(dst_idx, cur_idx,  xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;

    *xres = x + ax2;
    *yres = y + ay2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax-ax2, ay-ay2, bx, by);
  }

  if ((h2 & 1) && (h > 2)) {
    // prefer even steps
    bx2 += dbx;
    by2 += dby;
  }

  // standard case: one step up, one long horizontal, one step down
  nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, bx2,by2, ax2,ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, ax,ay, bx-bx2,by-by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return gilbert_d2xy_r(dst_idx, cur_idx,
                        xres,yres,
                        -bx2, -by2,
                        -(ax-ax2), -(ay-ay2));
}

int einsum_ir::backend::IterationSpacesSfc::gilbert_d3xy( int *x, 
                                                          int *y, 
                                                          int *z, 
                                                          int idx, 
                                                          int w, 
                                                          int h) {
  *x = 0;
  *y = 0;
  *z = idx/(w*h);
  idx = idx % (w*h);

  if (w >= h) {
    return gilbert_d2xy_r(idx,0, x,y, w,0, 0,h);
  }
  return gilbert_d2xy_r(idx,0, x,y, 0,h, w,0);
}


int64_t einsum_ir::backend::IterationSpacesSfc::num_tasks( ){
  return m_num_tasks;
}