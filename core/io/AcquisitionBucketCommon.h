#include "primitives.h"
#include "adapt_struct.h"
#include <mri_core_acquisition_bucket.h>

GADGETRON_ADAPT_STRUCT(AcquisitionBucketStats,
    GADGETRON_ACCESS_ELEMENT(kspace_encode_step_1),
    GADGETRON_ACCESS_ELEMENT(kspace_encode_step_2),
    GADGETRON_ACCESS_ELEMENT(slice),
    GADGETRON_ACCESS_ELEMENT(phase),
    GADGETRON_ACCESS_ELEMENT(contrast),
    GADGETRON_ACCESS_ELEMENT(repetition),
    GADGETRON_ACCESS_ELEMENT(set),
    GADGETRON_ACCESS_ELEMENT(segment),
    GADGETRON_ACCESS_ELEMENT(average)
)