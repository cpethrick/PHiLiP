
#include "ADTypes.hpp"

#include "euler_spacetime.h"

namespace PHiLiP {
namespace Physics {



#if PHILIP_DIM>1
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, double >;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, FadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, RadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, FadFadType>;
template class EulerSpacetime < PHILIP_DIM, PHILIP_DIM+2, RadFadType>;
#endif
} // Physics namespace
} // PHiLiP namespace
