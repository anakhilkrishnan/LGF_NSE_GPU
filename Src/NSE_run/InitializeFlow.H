#ifndef INITIALIZEFLOW_H_
#define INITIALIZEFLOW_H_

#include <FlowField.H>
#include <DirectSumLGF.H>

void initializeVelField(FlowField& init_state);
void initializePresField(FlowField& init_state, amrex::Real Re, amrex::Real source_tag_thresh);

#endif
