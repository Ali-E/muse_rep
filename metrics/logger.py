from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from scipy.stats import bootstrap
import numpy as np
import pandas as pd


class RougeEvalLogger:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []


    def log(self, prompt: str, gt: str, output: str, question: str | None = None):
        score = self.scorer.score(gt, output)
        d = {
            'prompt': prompt,
            'gt': gt,
            'response': output,
            'rougeL': score['rougeL'].fmeasure,
            'rougeL_recall': score['rougeL'].recall,
            'rouge1': score['rouge1'].fmeasure,
            'rouge1_recall': score['rouge1'].recall
        }
        if question is not None: d['question'] = question
        self.history.append(d)


    def report(self) -> Tuple[Dict, Dict]:
        agg = {}
        for key, val in self.history[0].items():
            if isinstance(val, str): continue
            vals: List[float] = [item[key] for item in self.history]
            agg[f"max_{key}"] = max(vals)
            agg[f"mean_{key}"] = sum(vals) / len(vals)
            agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = bootstrap(
                (vals,), np.mean,
                confidence_level=0.95,
                method='percentile'
            ).confidence_interval
        return agg, self.history


class RougeEvalLogger_new:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []


    def log(self, prompt: str, gt: str, stripped_output: str, output: str, question: str | None = None):
        score = self.scorer.score(gt, stripped_output)
        # d = {
        #     'prompt': prompt,
        #     'gt': gt,
        #     'response': stripped_output,
        #     'rougeL': score['rougeL'].fmeasure,
        #     'rougeL_recall': score['rougeL'].recall,
        #     'rouge1': score['rouge1'].fmeasure,
        #     'rouge1_recall': score['rouge1'].recall
        # }
        # if question is not None: d['question'] = question

        d = [prompt, gt, stripped_output, output,
             score['rougeL'].fmeasure, score['rougeL'].recall,
             score['rouge1'].fmeasure, score['rouge1'].recall]
        if question is not None: d.append(question) 

        self.history.append(d)
        # make a df and save to file:
        columns = [
            'prompt', 'expected_response', 'stripped_output', 'output',
            'rougeL', 'rougeL_recall',
            'rouge1', 'rouge1_recall'
        ]
        if len(self.history[0]) == 9:  # If question is included
            columns.append('question')
            columns_subset = ['question', 'expected_response', 'stripped_output', 'rougeL', 'rougeL_recall', 'rouge1', 'rouge1_recall']
        else:
            columns_subset = ['prompt', 'expected_response', 'stripped_output', 'rougeL', 'rougeL_recall', 'rouge1', 'rouge1_recall']

        history_df_short = pd.DataFrame(self.history, columns=columns)
        history_df_short = history_df_short[columns_subset]
        # history_df_short.to_csv('/scratch/aebrahim/muse_rep/temp/books_df_combined_short.csv', index=False)
        history_df_short.to_csv('/scratch/aebrahim/muse_rep/temp/books_df_orig_short.csv', index=False)
        

    def report(self) -> Tuple[Dict, Dict]:
        columns = [
            'prompt', 'expected_response', 'stripped_output', 'output',
            'rougeL', 'rougeL_recall',
            'rouge1', 'rouge1_recall'
        ]
        if len(self.history[0]) == 9:  # If question is included
            columns.append('question')
        history_df = pd.DataFrame(self.history, columns=columns)

        if len(self.history[0]) == 9:  # If question is included
            history_df = history_df[[
                'question','expected_response', 'stripped_output', 'rougeL', 'rougeL_recall',
                'rouge1', 'rouge1_recall'
            ]]

        agg = {}
        # compute aggregate statistics for rouge columns
        for key in [
            'rougeL',
            'rougeL_recall',
            'rouge1',
            'rouge1_recall'
        ]:
            vals: List[float] = history_df[key].tolist()
            agg[f"max_{key}"] = max(vals)
            agg[f"mean_{key}"] = sum(vals) / len(vals)
            ci = bootstrap(
                (vals,),
                np.mean,
                confidence_level=0.95,
                method='percentile'
            ).confidence_interval
            agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = ci

        return agg, history_df

