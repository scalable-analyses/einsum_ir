import os

print( 'running build script of einsum_ir' )

# configuration
l_vars = Variables()

l_vars.AddVariables(
  EnumVariable( 'mode',
                'compile modes, option \'san\' enables address and undefined behavior sanitizers',
                'release',
                 allowed_values=('release', 'debug', 'release+san', 'debug+san' )
              ),
  EnumVariable( 'parallel',
                'used parallelization',
                'omp',
                 allowed_values=('none', 'omp') ),
  PackageVariable( 'libxsmm',
                   'Enable libxsmm.',
                   'no' ),
  PackageVariable( 'libtorch',
                   'Enable libtorch.',
                   'no' )
)

# create environment
g_env = Environment( variables = l_vars )

# include environment
g_env['OS_ENV'] = os.environ

# exit in the case of unknown variables
if l_vars.UnknownVariables():
  print( "build configuration corrupted, don't know what to do with: " + str(l_vars.UnknownVariables().keys()) )
  exit(1)

# generate help message
Help( l_vars.GenerateHelpText(g_env) )

# configuration
g_conf = Configure( g_env )

# forward compiler
if 'CC' in g_env['OS_ENV'].keys():
  g_env['CC'] = g_env['OS_ENV']['CC']
if 'CXX' in g_env['OS_ENV'].keys():
  g_env['CXX'] = g_env['OS_ENV']['CXX']

# set optimization mode
if 'debug' in g_env['mode']:
  g_env.Append( CXXFLAGS = ['-g','-O0'] )
else:
  g_env.Append( CPPDEFINES = ['PP_NDEBUG'] )
  g_env.Append( CXXFLAGS = ['-O2'] )
# add sanitizers
if 'san' in  g_env['mode']:
  g_env.Append( CXXFLAGS =  ['-g', '-fsanitize=float-divide-by-zero', '-fsanitize=bounds', '-fsanitize=address', '-fsanitize=undefined', '-fno-omit-frame-pointer'] )
  g_env.Append( LINKFLAGS = ['-g', '-fsanitize=address', '-fsanitize=undefined'] )

# enable c++14
g_env.Append( CXXFLAGS = [ '-std=c++14' ] )

# enable omp
if 'omp' in g_env['parallel']:
  g_env.AppendUnique( CPPFLAGS = ['-fopenmp'] )
  g_env.AppendUnique( CPPFLAGS = ['-fopenmp-simd'] )
  g_env.AppendUnique( LINKFLAGS = ['-fopenmp'] )

# discover libraries
if g_env['libtorch'] != False:
  if g_env['libtorch'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libtorch'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['libtorch'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['libtorch'] + '/lib'] )
    g_env.AppendUnique( CPPDEFINES='_GLIBCXX_USE_CXX11_ABI=0' )

  if not( g_conf.CheckLib( 'libc10',
                           language='CXX' ) and \
          g_conf.CheckLib( 'libtorch_cpu',
                            language='CXX' ) and \
          g_conf.CheckLibWithHeader( 'libtorch',
                                      'ATen/ATen.h',
                                      'CXX' ) ):
    print( 'warning: disabling libtorch' )
    g_env['libtorch'] = False

if g_env['libxsmm'] != False:
  if g_env['libxsmm'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libxsmm'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['libxsmm'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['libxsmm'] + '/lib'] )

    if not g_conf.CheckLibWithHeader( 'libxsmm',
                                      'libxsmm.h',
                                      'CXX' ):
     print( 'warning: disabling libxsmm' )
     g_env['libxsmm'] = False

# build
g_env['build_dir'] = 'build'

g_env.AppendUnique( CPPPATH = [ '#.' ] )
g_env.AppendUnique( CPPPATH = [ '#/src' ] )

# get source files
VariantDir( g_env['build_dir']+'/src', 'src')

g_env.sources = []
g_env.tests = []
g_env.exe = {}

Export('g_env')
SConscript( g_env['build_dir']+'/src/SConscript' )
Import('g_env')

if( g_env['libxsmm'] and g_env['libtorch'] ):
  g_env.Program( g_env['build_dir']+'/bench_binary',
                 source = g_env.sources + g_env.exe['bench_binary'] )
  g_env.Program( g_env['build_dir']+'/bench_expression',
                 source = g_env.sources + g_env.exe['bench_expression'] )

g_env.Program( g_env['build_dir']+'/tests',
               source = g_env.tests )