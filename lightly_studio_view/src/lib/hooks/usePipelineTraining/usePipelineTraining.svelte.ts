import { createMutation, createQuery, useQueryClient } from '@tanstack/svelte-query';
import { toast } from 'svelte-sonner';
import type { StartTestRequest, TrainingRunView } from '$lib/api/lightly_studio_local';
import {
    deleteTrainingRunMutation,
    getTrainingLogOptions,
    getTrainingOptionsOptions,
    listTestDatasetsOptions,
    listTrainingRunsOptions,
    startTestRunMutation,
    startTrainingRunMutation,
    stopTrainingRunMutation
} from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { apiErrorMessage } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

const RUNS_KEY = [{ _id: 'listTrainingRuns' }];
const LOG_KEY = [{ _id: 'getTrainingLog' }];
const MODELS_KEY = [{ _id: 'listPipelineModels' }];
const OPTIONS_KEY = [{ _id: 'getTrainingOptions' }];
/** How often an active run is re-read. */
export const POLL_INTERVAL_MS = 3000;

/** True while the run trains or exports. */
export const isActiveRun = (run: Pick<TrainingRunView, 'status'>): boolean =>
    run.status === 'running' || run.status === 'exporting';

/** The v4 defaults, the model size presets and the environment check. */
export const useTrainingOptions = () => createQuery(() => getTrainingOptionsOptions());

/** The datasets that have a test split. */
export const useTestDatasets = () => createQuery(() => listTestDatasetsOptions());

/** "AP 82.1 · AP50 96.3 · 120 images · 3 seen in training" for a finished test. */
export const testResultLabel = (result: NonNullable<TrainingRunView['test_result']>): string => {
    const pct = (v: number | null) => (v === null ? '–' : (v * 100).toFixed(1));
    const parts = [
        `AP ${pct(result.ap50_95)}`,
        `AP50 ${pct(result.ap50)}`,
        `${result.images} images`
    ];
    if (result.overlap_images !== null && result.overlap_images > 0)
        parts.push(`${result.overlap_images} seen in training`);
    return parts.join(' · ');
};

/** All runs, newest first; polls while any run is active. */
export const useTrainingRuns = () =>
    createQuery(() => ({
        ...listTrainingRunsOptions(),
        refetchInterval: (query) => (query.state.data?.some(isActiveRun) ? POLL_INTERVAL_MS : false)
    }));

/** The log tail of one run, re-read while the run is active. */
export const useTrainingLog = (getParams: () => { name: string; active: boolean }) =>
    createQuery(() => {
        const { name, active } = getParams();
        return {
            ...getTrainingLogOptions({ path: { name }, query: { lines: 400 } }),
            refetchInterval: active ? POLL_INTERVAL_MS : false
        };
    });

/** Start, stop and delete runs; every change refreshes the list. */
export const useTrainingActions = () => {
    const client = useQueryClient();
    const refresh = () => {
        client.invalidateQueries({ queryKey: RUNS_KEY });
        client.invalidateQueries({ queryKey: LOG_KEY });
        client.invalidateQueries({ queryKey: MODELS_KEY });
        client.invalidateQueries({ queryKey: OPTIONS_KEY });
    };
    const onError = (fallback: string) => (error: unknown) =>
        toast.error(apiErrorMessage(error, fallback));

    const startMutation = createMutation(() => ({
        ...startTrainingRunMutation(),
        onSuccess: (run: TrainingRunView) => {
            refresh();
            toast.success(`Training "${run.name}" started on "${run.dataset}"`);
        },
        onError: onError('Starting the training failed')
    }));
    const testMutation = createMutation(() => ({
        ...startTestRunMutation(),
        onSuccess: (run: TrainingRunView) => {
            refresh();
            toast.success(`Test of "${run.model}" on "${run.dataset}" started`);
        },
        onError: onError('Starting the test failed')
    }));
    const stopMutation = createMutation(() => ({
        ...stopTrainingRunMutation(),
        onSuccess: (run: TrainingRunView) => {
            refresh();
            toast.success(`Training "${run.name}" stopped`);
        },
        onError: onError('Stopping failed')
    }));
    const deleteMutation = createMutation(() => ({
        ...deleteTrainingRunMutation(),
        onSuccess: () => {
            refresh();
            toast.success('Run deleted');
        },
        onError: onError('Deleting failed')
    }));

    return {
        startMutation,
        testMutation,
        stopMutation,
        deleteMutation,
        start: (body: Parameters<typeof startMutation.mutate>[0]['body']) =>
            startMutation.mutate({ body }),
        startTest: (body: StartTestRequest) => testMutation.mutate({ body }),
        stop: (name: string) => stopMutation.mutate({ path: { name } }),
        remove: (name: string) => deleteMutation.mutate({ path: { name } })
    };
};
