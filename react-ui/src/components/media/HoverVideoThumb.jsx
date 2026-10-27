import React from 'react';
import { Icon } from '../../icons';
import { useHoverMount, useVideoThumb } from './useHoverVideo';
import { releaseVideoElement } from './videoThumb';

/**
 * A video's picture in a list row: a still of its first frame, and a real <video> only while the pointer is over it.
 * (The Outputs tile does the same with scrubbing on top; see Gallery.jsx VideoHoverCard.)
 *
 * Pointer only: the tile holds no controls, so there is nothing for keyboard focus to land on.
 */
export default function HoverVideoThumb({ src, name = '', className = 'w-full h-full object-contain' }) {
  const still = useVideoThumb(src, true);
  const { on, bind } = useHoverMount();
  return (
    <div
      onPointerEnter={bind.onPointerEnter}
      onPointerLeave={bind.onPointerLeave}
      className="relative h-full w-full grid place-items-center"
      data-testid="hover-video-thumb"
    >
      {still ? (
        <img src={still} alt="" className={className} data-testid="video-still" />
      ) : (
        <span className="text-white/20" title={still === null ? `${name || 'video'}: no preview frame` : undefined}>
          <Icon.film size={22} />
        </span>
      )}
      {on && (
        <video
          ref={(el) => () => releaseVideoElement(el)}
          src={src}
          muted
          loop
          autoPlay
          playsInline
          preload="metadata"
          data-testid="hover-video"
          className={`absolute inset-0 bg-transparent pointer-events-none ${className}`}
          onError={(e) => { e.currentTarget.style.display = 'none'; }}
        />
      )}
    </div>
  );
}
