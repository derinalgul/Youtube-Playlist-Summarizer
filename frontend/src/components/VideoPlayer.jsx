import { useState, useEffect } from 'react';

export default function VideoPlayer({ citation }) {
  const [videoId, setVideoId] = useState(null);
  const [startTime, setStartTime] = useState(0);

  useEffect(() => {
    if (citation) {
      setVideoId(citation.video_id);
      setStartTime(Math.floor(citation.timestamp_seconds));
    }
  }, [citation]);

  if (!videoId) {
    return (
      <div className="video-player empty">
        <div className="placeholder">
          <p>Click a citation to watch the video</p>
        </div>
      </div>
    );
  }

  // Use YouTube embed with start time
  const embedUrl = `https://www.youtube.com/embed/${videoId}?start=${startTime}&autoplay=1`;

  return (
    <div className="video-player">
      <div className="video-container">
        <iframe
          key={`${videoId}-${startTime}`}
          src={embedUrl}
          title="YouTube video player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
      {citation && (
        <div className="video-info">
          <h3>{citation.video_title}</h3>
          <p className="timestamp">Starting at {citation.timestamp}</p>
          <p className="snippet">"{citation.text_snippet}"</p>
        </div>
      )}
    </div>
  );
}
