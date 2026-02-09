import { useState } from 'react';
import VideoSubmitForm from './components/VideoSubmitForm';
import JobStatus from './components/JobStatus';
import ChatInterface from './components/ChatInterface';
import VideoPlayer from './components/VideoPlayer';
import Stats from './components/Stats';

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [refreshStats, setRefreshStats] = useState(0);

  const handleJobCreated = (job) => {
    setJobs((prev) => [job, ...prev]);
  };

  const handleJobComplete = () => {
    setRefreshStats((prev) => prev + 1);
  };

  const handleCitationClick = (citation) => {
    setSelectedCitation(citation);
  };

  return (
    <div className="app">
      <header>
        <h1>YouTube Knowledge Base</h1>
        <Stats refreshTrigger={refreshStats} onCacheCleared={handleJobComplete} />
      </header>

      <main>
        <aside className="sidebar">
          <VideoSubmitForm onJobCreated={handleJobCreated} />

          {jobs.length > 0 && (
            <div className="jobs-section">
              <h3>Processing Jobs</h3>
              {jobs.map((job) => (
                <JobStatus
                  key={job.job_id}
                  job={job}
                  onComplete={handleJobComplete}
                />
              ))}
            </div>
          )}
        </aside>

        <section className="content">
          <div className="chat-section">
            <ChatInterface onCitationClick={handleCitationClick} />
          </div>

          <div className="video-section">
            <VideoPlayer citation={selectedCitation} />
          </div>
        </section>
      </main>
    </div>
  );
}
