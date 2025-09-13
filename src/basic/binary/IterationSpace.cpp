#include "IterationSpace.h"
#include "../third_party/gilbertSFC.cpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::basic::IterationSpace::init( std::vector< dim_t >   const * i_dim_types,
                                             std::vector< exec_t >  const * i_exec_types,
                                             std::vector< int64_t > const * i_sizes,
                                             int64_t                        i_num_threads_m,
                                             int64_t                        i_num_threads_n,
                                             int64_t                        i_num_threads_omp ) {
  m_dim_types  = i_dim_types;
  m_exec_types = i_exec_types;
  m_sizes      = i_sizes;
  m_num_threads_m   = i_num_threads_m;
  m_num_threads_n   = i_num_threads_n;
  m_num_threads_omp = i_num_threads_omp;
}

einsum_ir::basic::err_t einsum_ir::basic::IterationSpace::setup( std::vector< int64_t >   & io_strides_left,
                                                                 std::vector< int64_t >   & io_strides_right,
                                                                 std::vector< int64_t >   & io_strides_out_aux,
                                                                 std::vector< int64_t >   & io_strides_out, 
                                                                 std::vector<thread_info> & io_thread_infos ) {
                
  //calculate number of tasks and assigns parallel dimensions to three types omp, sfc_n, sfc_m
  m_sfc_tasks_m = 1;
  m_sfc_tasks_n = 1;
  m_sfc_tasks_k = 1;
  m_omp_tasks   = 1;
  bool l_found_omp = false;
  int64_t l_num_sfc_loops = 0;
  int64_t l_num_omp_loops = 0;
  int64_t l_last_found_sfc_type = 0;
  for( std::size_t l_id = 0; l_id < m_dim_types->size(); l_id++ ){
    if( m_exec_types->at(l_id) == exec_t::OMP ){
      l_num_omp_loops++;
      if( l_found_omp == false ){
        m_omp_loops.begin = l_id;
        l_found_omp = true;
      }
      m_omp_tasks *= m_sizes->at(l_id);
      m_omp_loops.end = l_id + 1;
      if( m_dim_types->at(l_id) == dim_t::K ){
        return err_t::COMPILATION_FAILED;
      }
    }
    if( m_exec_types->at(l_id) == exec_t::SFC ){
      l_num_sfc_loops++;
      if(m_dim_types->at(l_id)  == dim_t::M) {
        if( l_last_found_sfc_type <= 1 ){
          m_sfc_loops_m.begin = l_id;
          l_last_found_sfc_type = 2;
        }
        m_sfc_tasks_m *= m_sizes->at(l_id);
        m_sfc_loops_m.end = l_id + 1;  
      }
      else if(m_dim_types->at(l_id)  == dim_t::N ) {
        if( l_last_found_sfc_type <= 2 ){
          m_sfc_loops_n.begin = l_id;
          l_last_found_sfc_type = 3;
        }
        m_sfc_tasks_n *= m_sizes->at(l_id);
        m_sfc_loops_n.end = l_id + 1;
      }
      else if(m_dim_types->at(l_id)  == dim_t::K ) {
        if( l_last_found_sfc_type <= 3 ){
          m_sfc_loops_k.begin = l_id;
          l_last_found_sfc_type = 4;
        }
        m_sfc_tasks_k *= m_sizes->at(l_id);
        m_sfc_loops_k.end = l_id + 1;
      }
      else{
        return err_t::COMPILATION_FAILED;
      }
    }
  }

  //check that all parallel loops are next to each other
  if( m_sfc_loops_m.end - m_sfc_loops_m.begin +
      m_sfc_loops_n.end - m_sfc_loops_n.begin +
      m_sfc_loops_k.end - m_sfc_loops_k.begin != 
      l_num_sfc_loops ){
    return err_t::COMPILATION_FAILED;
  }
  if( m_omp_loops.end - m_omp_loops.begin != 
      l_num_omp_loops ){
    return err_t::COMPILATION_FAILED;
  }

  //create thread infos
  int64_t l_num_threads = m_num_threads_m * m_num_threads_n * m_num_threads_omp;
  io_thread_infos.resize( l_num_threads );
  m_tasks_per_thread_m   = (m_sfc_tasks_m + m_num_threads_m   - 1) / m_num_threads_m;
  m_tasks_per_thread_n   = (m_sfc_tasks_n + m_num_threads_n   - 1) / m_num_threads_n;
  m_tasks_per_thread_omp = (m_omp_tasks   + m_num_threads_omp - 1) / m_num_threads_omp;
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(l_num_threads)
#endif
  for( int64_t l_thread_id = 0; l_thread_id < l_num_threads; l_thread_id++ ){
    //get thread id in m and n 
    int64_t l_thread_id_m   = l_thread_id % m_num_threads_m;
    int64_t l_rem           = l_thread_id / m_num_threads_m;
    int64_t l_thread_id_n   = l_rem % m_num_threads_n;
    int64_t l_thread_id_omp = l_rem / m_num_threads_n;

    //calculate begin and end for sfc m dimension 
    int64_t l_begin_m, l_end_m;
    l_begin_m = l_thread_id_m * m_tasks_per_thread_m;
    l_end_m   = l_begin_m     + m_tasks_per_thread_m;
    l_begin_m = l_begin_m <= m_sfc_tasks_m ? l_begin_m : m_sfc_tasks_m;
    l_end_m   = l_end_m   <= m_sfc_tasks_m ? l_end_m   : m_sfc_tasks_m;

    //calculate begin and end for sfc n dimension 
    int64_t l_begin_n, l_end_n;
    l_begin_n = l_thread_id_n * m_tasks_per_thread_n;
    l_end_n   = l_begin_n     + m_tasks_per_thread_n;
    l_begin_n = l_begin_n <= m_sfc_tasks_n ? l_begin_n : m_sfc_tasks_n;
    l_end_n   = l_end_n   <= m_sfc_tasks_n ? l_end_n   : m_sfc_tasks_n;

    //calculate begin and end for omp dimension
    int64_t l_begin_omp, l_end_omp;
    l_begin_omp = l_thread_id_omp * m_tasks_per_thread_omp;
    l_end_omp   = l_begin_omp     + m_tasks_per_thread_omp;
    l_begin_omp = l_begin_omp <= m_omp_tasks ? l_begin_omp : m_omp_tasks;
    l_end_omp   = l_end_omp   <= m_omp_tasks ? l_end_omp   : m_omp_tasks;
    io_thread_infos[l_thread_id].id_omp_loop_start = l_begin_omp;
    io_thread_infos[l_thread_id].id_omp_loop_end   = l_end_omp;
    
    //set start ids
    int64_t l_id_sfc_m_old = l_begin_m;
    int64_t l_id_sfc_n_old = l_begin_n;
    int64_t l_id_sfc_k_old = 0;

    //set thread properties
    io_thread_infos[l_thread_id].sfc_size_m = l_end_m - l_begin_m;
    io_thread_infos[l_thread_id].sfc_size_n = l_end_n - l_begin_n;
    io_thread_infos[l_thread_id].sfc_size_k = m_sfc_tasks_k;
    io_thread_infos[l_thread_id].k_count.resize( (l_end_m - l_begin_m) *
                                                 (l_end_n - l_begin_n), 0 );

    //calculate initial thread offsets
    int64_t l_offset;
    l_offset = calculate_offset( l_id_sfc_m_old, l_id_sfc_n_old, io_strides_left );
    io_thread_infos[l_thread_id].offset_left = l_offset;
    l_offset = calculate_offset( l_id_sfc_m_old, l_id_sfc_n_old, io_strides_right );
    io_thread_infos[l_thread_id].offset_right = l_offset;
    l_offset = calculate_offset( l_id_sfc_m_old, l_id_sfc_n_old, io_strides_out );
    io_thread_infos[l_thread_id].offset_out = l_offset;
    l_offset = calculate_offset( l_id_sfc_m_old, l_id_sfc_n_old, io_strides_out_aux);
    io_thread_infos[l_thread_id].offset_out_aux = l_offset;

    //calculate movements
    int64_t l_size = (l_end_m - l_begin_m) * (l_end_n - l_begin_n) * m_sfc_tasks_k;
    io_thread_infos[l_thread_id].movement_ids.resize( l_size );
    for( int64_t l_id = 0; l_id < l_size; l_id++ ){
      //determine new SFC position
      int64_t l_id_sfc_m_new, l_id_sfc_n_new, l_id_sfc_k_new; 
      sfc_oracle_3d( &l_id_sfc_m_new, &l_id_sfc_n_new, &l_id_sfc_k_new, l_id+1, l_end_m - l_begin_m, l_end_n - l_begin_n, m_sfc_tasks_k );
      l_id_sfc_m_new += l_begin_m;
      l_id_sfc_n_new += l_begin_n;

      //determine movement
      if( l_id_sfc_m_new != l_id_sfc_m_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old );
        io_thread_infos[l_thread_id].movement_ids[l_id] = l_move;
      }
      else if( l_id_sfc_n_new != l_id_sfc_n_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old );
        io_thread_infos[l_thread_id].movement_ids[l_id] = l_move;
      }
      else if( l_id_sfc_k_new != l_id_sfc_k_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_k, l_id_sfc_k_new, l_id_sfc_k_old );
        io_thread_infos[l_thread_id].movement_ids[l_id] = l_move;
      }

      l_id_sfc_m_old = l_id_sfc_m_new;
      l_id_sfc_n_old = l_id_sfc_n_new;
      l_id_sfc_k_old = l_id_sfc_k_new;
    }
  }

  //convert strides to offsets
  convert_strides_to_offsets( io_strides_left    );
  convert_strides_to_offsets( io_strides_right   );
  convert_strides_to_offsets( io_strides_out     );
  convert_strides_to_offsets( io_strides_out_aux );

  return err_t::SUCCESS;
}

int64_t einsum_ir::basic::IterationSpace::calculate_offset( int64_t i_id_sfc_m,
                                                            int64_t i_id_sfc_n,
                                                            std::vector< int64_t > const & i_strides ) {

  int64_t l_offset = 0;
  for (int64_t l_id = m_sfc_loops_m.end - 1; l_id >= m_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_m % l_size) * l_stride;
    i_id_sfc_m /= l_size;
  }
  for (int64_t l_id = m_sfc_loops_n.end - 1; l_id >= m_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_n % l_size) * l_stride;
    i_id_sfc_n /= l_size;
  }

  return l_offset;
}

void einsum_ir::basic::IterationSpace::convert_strides_to_offsets( std::vector< int64_t > & io_strides) {

  int64_t l_id_sfc_m, l_id_sfc_n, l_id_omp;
  int64_t l_all_offsets_sfc_m = 0;
  int64_t l_all_offsets_sfc_n = 0;
  int64_t l_all_offsets_sfc_k = 0;

  for (int64_t l_id = m_sfc_loops_m.end - 1; l_id >= m_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = io_strides[l_id];
    io_strides[l_id] = l_stride - l_all_offsets_sfc_m;
    l_all_offsets_sfc_m += (l_size - 1) * l_stride;
  }
  
  for (int64_t l_id = m_sfc_loops_n.end - 1; l_id >= m_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = io_strides[l_id];
    io_strides[l_id] = l_stride - l_all_offsets_sfc_n;
    l_all_offsets_sfc_n += (l_size - 1) * l_stride;
  }

  for (int64_t l_id = m_sfc_loops_k.end - 1; l_id >= m_sfc_loops_k.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = io_strides[l_id];
    io_strides[l_id] = l_stride - l_all_offsets_sfc_k;
    l_all_offsets_sfc_k += (l_size - 1) * l_stride;
  }
}

einsum_ir::basic::sfc_t einsum_ir::basic::IterationSpace::get_max_dim_jump( range_t i_dim_loops,
                                                                            int64_t i_id_new,
                                                                            int64_t i_id_old ){

  int64_t l_direction = (( i_id_old - i_id_new ) + 1) >> 1;
  int64_t l_max_id = i_id_new + l_direction;
  for( int64_t l_di = i_dim_loops.end-1; l_di >= i_dim_loops.begin; l_di-- ){
    int64_t l_size = m_sizes->at(l_di);
    int64_t l_rem  = l_max_id % l_size;
    l_max_id       = l_max_id / l_size;

    if(l_rem != 0){
      return (l_di << 1) + l_direction;
    }
  }

  return 0;
}


void einsum_ir::basic::IterationSpace::sfc_oracle_2d( int64_t *o_m, 
                                                      int64_t *o_n,
                                                      int64_t  i_idx,
                                                      int64_t  i_sfc_size_m,
                                                      int64_t  i_sfc_size_n ) {
  
  int l_w = i_sfc_size_m;
  int l_h = i_sfc_size_n;

  int l_idx_m, l_idx_n;
  gilbert_d2xy(&l_idx_m, &l_idx_n, i_idx, l_w, l_h);

  *o_m = l_idx_m;
  *o_n = l_idx_n;
}

int64_t einsum_ir::basic::IterationSpace::get_caching_size(){

  //caching more entrys than the number of task in one dimension won't be useful
  int64_t l_max_cache_size = std::min(m_tasks_per_thread_m, m_tasks_per_thread_n);

  //upper bound to prevent huge memory allocations
  l_max_cache_size = std::min(l_max_cache_size, (int64_t)8);

  return l_max_cache_size;
}


void einsum_ir::basic::IterationSpace::sfc_oracle_3d( int64_t *o_m, 
                                                      int64_t *o_n,
                                                      int64_t *o_k,
                                                      int64_t  i_idx,
                                                      int64_t  i_sfc_size_m,
                                                      int64_t  i_sfc_size_n,
                                                      int64_t  i_sfc_size_k ) {

  int l_w = i_sfc_size_m;
  int l_h = i_sfc_size_n;
  int l_d = i_sfc_size_k;

  int l_idx_m_n, l_idx_m, l_idx_n, l_idx_k;
  gilbert_d2xy(&l_idx_m_n, &l_idx_k, i_idx, l_w*l_h, l_d);
  gilbert_d2xy(&l_idx_m, &l_idx_n, l_idx_m_n, l_w, l_h);

  *o_k = l_idx_k;
  *o_m = l_idx_m;
  *o_n = l_idx_n;
}
