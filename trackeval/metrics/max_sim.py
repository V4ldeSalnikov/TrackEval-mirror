
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class MaxSim(_BaseMetric):
    """Class which implements the MaxSim metric between  GT and prediction """

    def __init__(self):
        super().__init__()
        self.integer_fields = ['MaxSim_Detections']
        self.float_fields = ['MaxSim']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates MaxSim metric for one sequence"""
        res = {
            'MaxSim': 0.0,
            'MaxSim_Detections': 0
        }

        # if no detection found
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
            return res

        # Find maximum similarity across all timestemps
        max_similarity = 0.0
        for t in range(data['num_timesteps']):
            if data['similarity_scores'][t].size > 0:
                timestep_max = np.max(data['similarity_scores'][t])
                max_similarity = max(max_similarity, timestep_max)

        res['MaxSim'] = max_similarity
        res['MaxSim_Detections'] = data['num_tracker_dets']
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences by averaging"""
        res = {}
        res['MaxSim'] = np.mean([v['MaxSim'] for v in all_res.values()])
        res['MaxSim_Detections'] = self._combine_sum(all_res, 'MaxSim_Detections')
        return res

    def combine_classes_class_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the class values """
        res = {}

        valid_classes = [v for v in all_res.values() if v['MaxSim_Detections'] > 0]
        if len(valid_classes) > 0:
            res['MaxSim'] = np.mean([v['MaxSim'] for v in valid_classes])
        else:
            res['MaxSim'] = 0.0

        res['MaxSim_Detections'] = self._combine_sum(all_res, 'MaxSim_Detections')

        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes weighted by averaging over the detection values"""
        res = {}
        total_detections = self._combine_sum(all_res, 'MaxSim_Detections')
        res['MaxSim_Detections'] = total_detections

        #Calculatiuon of weighted average
        if total_detections > 0:
            weighted_sum = sum([all_res[k]['MaxSim'] * all_res[k]['MaxSim_Detections']
                                for k in all_res.keys()])
            res['MaxSim'] = weighted_sum / total_detections
        else:
            res['MaxSim'] = 0.0


        return res

