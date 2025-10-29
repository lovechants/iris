use ratatui::{prelude::*, widgets::*};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
};
use serde::Deserialize;
use std::{fs, io, thread::sleep, time::Duration, path::PathBuf};

#[derive(Deserialize, Clone)]
struct Record {
    timestamp: f64,
    kernel: String,
    time_ms: f64,
    phase: String,
}

fn read_records(path: &PathBuf) -> Vec<Record> {
    if let Ok(data) = fs::read_to_string(path) {
        data.lines()
            .filter_map(|l| serde_json::from_str::<Record>(l).ok())
            .collect()
    } else {
        vec![]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let home = dirs::home_dir().unwrap();
    let log_path = home.join(".iris_cache/iris_log.jsonl");

    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('r') => {
                        if fs::write(&log_path, "").is_ok() {}
                    }
                    _ => {}
                }
            }
        }

        let records = read_records(&log_path);
        let count = records.len();
        let recent: Vec<_> = records.iter().rev().take(20).cloned().collect();

        terminal.draw(|f| {
            let size = f.size();
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [Constraint::Length(2), Constraint::Min(0), Constraint::Length(1)].as_ref(),
                )
                .split(size);

            // header
            let header_text = format!(
                " Iris GPU Runtime | q=quit | r=reset | last records: {:<3} ",
                count
            );
            let header = Paragraph::new(header_text)
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
            f.render_widget(header, layout[0]);

            // table
            let rows: Vec<Row> = recent
                .iter()
                .map(|r| {
                    let (color, label) = match r.phase.as_str() {
                        "compile" => (Color::Yellow, "Compile"),
                        "run" | "gpu_run" => (Color::Green, "Run"),
                        "hit" => (Color::DarkGray, "Cache"),
                        _ => (Color::Gray, "Other"),
                    };

                    let bar_len = ((r.time_ms * 5.0).min(60.0)) as usize;
                    let bar = "█".repeat(bar_len.max(1));

                    Row::new(vec![
                        Cell::from(r.kernel.clone()).style(Style::default().fg(color)),
                        Cell::from(format!("{:>8.3}", r.time_ms)),
                        Cell::from(bar).style(Style::default().fg(color)),
                        Cell::from(label).style(Style::default().fg(Color::Gray)),
                    ])
                })
                .collect();

            let table = Table::new(
                rows,
                [
                    Constraint::Length(25),
                    Constraint::Length(10),
                    Constraint::Length(20),
                    Constraint::Min(8),
                ],
            )
            .header(
                Row::new(vec!["Kernel", "Time(ms)", "Bar", "Phase"])
                    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            )
            .block(Block::default().borders(Borders::ALL).title(" GPU Activity "))
            .column_spacing(1);

            f.render_widget(table, layout[1]);

            // footer
            if count > 0 {
                let avg: f64 = records.iter().map(|r| r.time_ms).sum::<f64>() / count as f64;
                let footer = Paragraph::new(format!(
                    " total entries: {:<5} | avg time: {:>8.3} ms ",
                    count, avg
                ))
                .style(Style::default().fg(Color::DarkGray));
                f.render_widget(footer, layout[2]);
            }
        })?;

        sleep(Duration::from_millis(100));
    }

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}
