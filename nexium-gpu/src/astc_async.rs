use std::sync::mpsc;

pub(crate) struct DecodeWorker<J, R> {
    jobs: mpsc::Sender<J>,
    results: mpsc::Receiver<R>,
    in_flight: usize,
}

impl<J: Send + 'static, R: Send + 'static> DecodeWorker<J, R> {
    pub(crate) fn spawn(name: &str, work: impl Fn(J) -> R + Send + 'static) -> Result<Self, String> {
        let (jobs, job_receiver) = mpsc::channel::<J>();
        let (result_sender, results) = mpsc::channel::<R>();
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                for job in job_receiver {
                    if result_sender.send(work(job)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("{name}: {error}"))?;
        Ok(Self {
            jobs,
            results,
            in_flight: 0,
        })
    }

    pub(crate) fn submit(&mut self, job: J) -> bool {
        let sent = self.jobs.send(job).is_ok();
        self.in_flight += usize::from(sent);
        sent
    }

    pub(crate) fn completed(&mut self) -> Vec<R> {
        let results: Vec<R> = self.results.try_iter().collect();
        self.in_flight = self.in_flight.saturating_sub(results.len());
        results
    }

    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeWorker;

    #[test]
    fn worker_returns_every_job_in_order() {
        let mut worker = DecodeWorker::spawn("test-decode", |value: u32| value * 2).unwrap();
        for value in 0..64 {
            assert!(worker.submit(value));
        }
        let mut results = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while results.len() < 64 && std::time::Instant::now() < deadline {
            results.extend(worker.completed());
            std::thread::yield_now();
        }
        assert_eq!(results, (0..64).map(|value| value * 2).collect::<Vec<_>>());
        assert_eq!(worker.in_flight(), 0);
    }
}
