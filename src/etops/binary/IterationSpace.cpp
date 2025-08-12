#include "IterationSpace.h"
#include "../third_party/gilbertSFC.cpp"

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::etops::IterationSpace::init( std::vector< dim_t >   const * i_dim_types,
                                             std::vector< exec_t >  const * i_exec_types,
                                             std::vector< int64_t > const * i_sizes,
                                             int64_t                        i_num_threads){
  m_dim_types  = i_dim_types;
  m_exec_types = i_exec_types;
  m_sizes      = i_sizes;
  m_num_threads = i_num_threads;
}

einsum_ir::etops::err_t einsum_ir::etops::IterationSpace::setup( std::vector< int64_t >   & io_strides_left,
                                                                 std::vector< int64_t >   & io_strides_right,
                                                                 std::vector< int64_t >   & io_strides_out_aux,
                                                                 std::vector< int64_t >   & io_strides_out, 
                                                                 std::vector<thread_info> & io_thread_infos ) {
                
  //calculate number of tasks and assigns parallel dimensions to three types omp, sfc_n, sfc_m
  m_sfc_tasks_m = 1;
  m_sfc_tasks_n = 1;
  int64_t l_num_tasks = 1;
  int64_t l_num_parallel_loops = 0;
  int64_t l_last_found_type = 0;
  for( std::size_t l_id = 0; l_id < m_dim_types->size(); l_id++ ){
    if( m_exec_types->at(l_id) == exec_t::OMP ||
        m_exec_types->at(l_id) == exec_t::SFC    ){
      l_num_tasks *= m_sizes->at(l_id);
      l_num_parallel_loops += 1;

      if(m_exec_types->at(l_id) == exec_t::OMP) {
        if( l_last_found_type == 0 ){
          m_omp_loops.begin = l_id;
          l_last_found_type = 1;
        }
        m_omp_loops.end = l_id + 1;
      }
      else if(m_dim_types->at(l_id)  == dim_t::M) {
        if( l_last_found_type <= 1 ){
          m_sfc_loops_m.begin = l_id;
          l_last_found_type = 2;
        }
        m_sfc_tasks_m *= m_sizes->at(l_id);
        m_sfc_loops_m.end = l_id + 1;  
      }
      else if(m_dim_types->at(l_id)  == dim_t::N ) {
        if( l_last_found_type <= 2 ){
          m_sfc_loops_n.begin = l_id;
          l_last_found_type = 3;
        }
        m_sfc_tasks_n *= m_sizes->at(l_id);
        m_sfc_loops_n.end = l_id + 1;
      }
      else{
        return err_t::COMPILATION_FAILED;
      }
    }
  }

  //check that all parallel loops are next to each other
  if( m_omp_loops.end   - m_omp_loops.begin   +
      m_sfc_loops_m.end - m_sfc_loops_m.begin +
      m_sfc_loops_n.end - m_sfc_loops_n.begin !=
      l_num_parallel_loops ){
    return err_t::COMPILATION_FAILED;
  }

  //create thread infos 
  io_thread_infos.resize( m_num_threads );
  int64_t l_tasks_per_thread = l_num_tasks / m_num_threads + (l_num_tasks % m_num_threads != 0);
  
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_num_threads)
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_id * l_tasks_per_thread;
    int64_t l_end   = l_begin     + l_tasks_per_thread;
    l_begin = l_begin < l_num_tasks ? l_begin : l_num_tasks;
    l_end   = l_end   < l_num_tasks ? l_end   : l_num_tasks;

    int64_t l_id_sfc_m_old, l_id_sfc_n_old, l_id_omp_old;
    sfc_oracle_2d( &l_id_sfc_m_old, &l_id_sfc_n_old, &l_id_omp_old, l_begin );

    //calculate initial thread offsets
    int64_t l_offset;
    l_offset = calculate_offset( l_id_omp_old, l_id_sfc_m_old, l_id_sfc_n_old, io_strides_left );
    io_thread_infos[l_thread_id].offset_left = l_offset;

    l_offset = calculate_offset( l_id_omp_old, l_id_sfc_m_old, l_id_sfc_n_old, io_strides_right );
    io_thread_infos[l_thread_id].offset_right = l_offset;

    l_offset = calculate_offset( l_id_omp_old, l_id_sfc_m_old, l_id_sfc_n_old, io_strides_out );
    io_thread_infos[l_thread_id].offset_out = l_offset;

    l_offset = calculate_offset( l_id_omp_old, l_id_sfc_m_old, l_id_sfc_n_old, io_strides_out_aux);
    io_thread_infos[l_thread_id].offset_out_aux = l_offset;

    //calculate movements
    io_thread_infos[l_thread_id].movement_ids.resize( l_end - l_begin );
    for( int64_t l_id = l_begin; l_id < l_end; l_id++ ){
      int64_t l_id_sfc_m_new, l_id_sfc_n_new, l_id_omp_new; 
      sfc_oracle_2d( &l_id_sfc_m_new, &l_id_sfc_n_new, &l_id_omp_new, l_id+1 );

      if( l_id_omp_new != l_id_omp_old ){
        sfc_t l_move = get_max_dim_jump( m_omp_loops, l_id_omp_new, l_id_omp_old );
        io_thread_infos[l_thread_id].movement_ids[l_id-l_begin] = l_move; 
      }
      else if( l_id_sfc_m_new != l_id_sfc_m_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old );
        io_thread_infos[l_thread_id].movement_ids[l_id-l_begin] = l_move;
      }
      else if( l_id_sfc_n_new != l_id_sfc_n_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old );
        io_thread_infos[l_thread_id].movement_ids[l_id-l_begin] = l_move;
      }

      l_id_sfc_m_old = l_id_sfc_m_new;
      l_id_sfc_n_old = l_id_sfc_n_new;
      l_id_omp_old = l_id_omp_new;
    }
  }

  //convert strides to offsets
  convert_strides_to_offsets( io_strides_left    );
  convert_strides_to_offsets( io_strides_right   );
  convert_strides_to_offsets( io_strides_out     );
  convert_strides_to_offsets( io_strides_out_aux );

  return err_t::SUCCESS;
}

int64_t einsum_ir::etops::IterationSpace::calculate_offset( int64_t i_id_omp,
                                                            int64_t i_id_sfc_m,
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
  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_omp % l_size) * l_stride;
    i_id_omp /= l_size;
  }

  return l_offset;
}

void einsum_ir::etops::IterationSpace::convert_strides_to_offsets( std::vector< int64_t > & io_strides) {

  int64_t l_id_sfc_m, l_id_sfc_n, l_id_omp;
  sfc_oracle_2d(&l_id_sfc_m, &l_id_sfc_n, &l_id_omp, m_sfc_tasks_m*m_sfc_tasks_n-1);
  int64_t l_all_offsets_omp   = calculate_offset( l_id_omp, l_id_sfc_m, l_id_sfc_n, io_strides );
  int64_t l_all_offsets_sfc_m = 0;
  int64_t l_all_offsets_sfc_n = 0;

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

  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_sizes->at(l_id);
    int64_t l_stride = io_strides[l_id];

    io_strides[l_id] = l_stride - l_all_offsets_omp;
    l_all_offsets_omp += (l_size - 1) * l_stride;
  }
}

einsum_ir::etops::sfc_t einsum_ir::etops::IterationSpace::get_max_dim_jump( range_t i_dim_loops,
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

void einsum_ir::etops::IterationSpace::sfc_oracle_2d( int64_t *o_m, 
                                                      int64_t *o_n,
                                                      int64_t *o_omp, 
                                                      int64_t  i_idx ){
  
  int l_w = m_sfc_tasks_m;
  int l_h = m_sfc_tasks_n;
  *o_omp = i_idx / (l_w*l_h);
  i_idx = i_idx % (l_w*l_h);

  int l_idx_m, l_idx_n;
  gilbert_d2xy(&l_idx_m, &l_idx_n, i_idx, l_w, l_h);

  *o_m = l_idx_m;
  *o_n = l_idx_n;
}
