import React, { useState } from 'react';
import Viewport from './Viewport';
import Timeline from './Timeline';
import RunDeck from './RunDeck';
import ToolsPanel from './Tools';
import { CastPanel, TargetsPanel, PeoplePanel, PipelinePanel, LookPanel } from './Panels';
import { Panel } from '../fx/effects';
import { useStudio, currentTarget } from '../engine/studio';

// The studio: cast and media on the left, the viewport and its timeline in the middle, the people,
// render and pipeline on the right. Stacks to one column under 1280px.
export default function Studio({ onOpenLibrary }) {
  const [scrubbing, setScrubbing] = useState(false);
  const target = useStudio(currentTarget);
  return (
    <div className="grid gap-3 xl:h-full xl:grid-cols-[280px_minmax(0,1fr)_340px] 3xl:grid-cols-[320px_minmax(0,1fr)_380px]" data-testid="studio">
      <div className="flex min-h-0 flex-col gap-3 xl:overflow-y-auto xl:pr-1">
        <CastPanel onOpenLibrary={onOpenLibrary} />
        <TargetsPanel />
      </div>
      <div className="flex min-h-0 min-w-0 flex-col gap-3">
        <div className="h-[56vh] min-h-[280px] xl:h-auto xl:min-h-[300px] xl:flex-1">
          <Viewport scrubbing={scrubbing} />
        </div>
        {target && (
          <Panel bodyClass="px-3 pb-3 pt-2.5" data-testid="timeline-panel">
            <Timeline onScrub={setScrubbing} />
          </Panel>
        )}
      </div>
      <div className="flex min-h-0 flex-col gap-3 xl:overflow-y-auto xl:pr-1">
        <RunDeck />
        <PeoplePanel />
        <ToolsPanel />
        <PipelinePanel />
        <LookPanel />
      </div>
    </div>
  );
}
