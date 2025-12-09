#ifndef GEQRF_BATCHED_PANEL_DECISION
#define GEQRF_BATCHED_PANEL_DECISION

#define GEQRF_BATCHED_LOOKUP_TABLE_BATCH_STEP   (100)
#define GEQRF_BATCHED_MAX_TESTED_WIDTH          (256)

#ifdef MAGMA_HAVE_HIP
#include "geqrf_panel_decision_mi100.h"
#include "geqrf_panel_decision_mi300a.h"
#else
#include "geqrf_panel_decision_a100.h"
#include "geqrf_panel_decision_h100.h"
#endif

#endif    // GEQRF_BATCHED_PANEL_DECISION
