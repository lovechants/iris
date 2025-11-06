use ratatui::{prelude::*, widgets::*};
use crossterm::{
    event::{self, Event, KeyCode},
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
};
use serde::Deserialize;
use std::{fs, io, thread::sleep, time::Duration, path::PathBuf, collections::HashMap};

#[derive(Deserialize, Clone)]
struct Record {
    timestamp: f64,
    kernel: String,
    time_ms: f64,
    phase: String,
    memory_usage: Option<f64>,
    thread_id: Option<u32>,
    kernel_type: Option<String>, 
}

struct App {
    records: Vec<Record>,
    selected_kernel: Option<String>,
    view_mode: ViewMode,
    sort_mode: SortMode,
    filter_phase: Option<String>,
    time_range: TimeRange,
    stats: HashMap<String, KernelStats>,
}

#[derive(Default)]
struct KernelStats {
    total_time: f64,
    call_count: u32,
    avg_time: f64,
    min_time: f64,
    max_time: f64,
}

#[derive(Default, Clone, Copy, Debug)] 
enum ViewMode {
    #[default]
    Timeline,
    Statistics,
    KernelDetails,
    MemoryUsage,
}

#[derive(Default, Clone, Copy, Debug)] 
enum SortMode {
    #[default]
    ByTime,
    ByName,
    ByFrequency,
}

#[derive(Default, Clone)]
struct TimeRange {
    start: Option<f64>,
    end: Option<f64>,
}

impl App {
    fn new() -> Self {
        Self {
            records: vec![],
            selected_kernel: None,
            view_mode: ViewMode::default(),
            sort_mode: SortMode::default(),
            filter_phase: None,
            time_range: TimeRange::default(),
            stats: HashMap::new(),
        }
    }
    
    fn update_stats(&mut self) {
        self.stats.clear();
        
        for record in &self.records {
            let stats = self.stats.entry(record.kernel.clone()).or_default();
            stats.total_time += record.time_ms;
            stats.call_count += 1;
            stats.avg_time = stats.total_time / stats.call_count as f64;
            stats.min_time = stats.min_time.min(record.time_ms);
            stats.max_time = stats.max_time.max(record.time_ms);
        }
    }
    
    fn filtered_records(&self) -> Vec<&Record> {
        self.records.iter()
            .filter(|r| {
                // Apply phase filter
                if let Some(phase) = &self.filter_phase {
                    if r.phase != *phase {
                        return false;
                    }
                }
                
                // Apply time range filter
                if let (Some(start), Some(end)) = (self.time_range.start, self.time_range.end) {
                    if r.timestamp < start || r.timestamp > end {
                        return false;
                    }
                }
                
                true
            })
            .collect()
    }
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

fn ui(f: &mut Frame, app: &mut App) {
    let area = f.area(); 
    
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Footer/Help
        ])
        .split(area);
    
    render_header(f, chunks[0], app);
    render_content(f, chunks[1], app);
    render_footer(f, chunks[2], app);
}

fn render_header(f: &mut Frame, area: Rect, app: &App) {
    let header_text = format!(
        " Iris GPU Profiler | Mode: {:?} | Sort: {:?} | q=quit | r=reset | 1-4=views | s=sort | f=filter ",
        app.view_mode, app.sort_mode
    );
    
    let header = Paragraph::new(header_text)
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(header, area);
}

fn render_content(f: &mut Frame, area: Rect, app: &mut App) {
    match app.view_mode {
        ViewMode::Timeline => render_timeline_view(f, area, app),
        ViewMode::Statistics => render_statistics_view(f, area, app),
        ViewMode::KernelDetails => render_kernel_details_view(f, area, app),
        ViewMode::MemoryUsage => render_memory_view(f, area, app),
    }
}

fn render_timeline_view(f: &mut Frame, area: Rect, app: &App) {
    let filtered = app.filtered_records();
    let recent: Vec<_> = filtered.iter().rev().take(area.height as usize - 4).collect();

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

            let kernel_name = if let Some(selected) = &app.selected_kernel {
                if r.kernel == *selected {
                    format!("→ {}", r.kernel)
                } else {
                    r.kernel.clone()
                }
            } else {
                r.kernel.clone()
            };

            Row::new(vec![
                Cell::from(kernel_name).style(Style::default().fg(color)),
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
    .block(Block::default().borders(Borders::ALL).title(" GPU Activity Timeline "))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn render_statistics_view(f: &mut Frame, area: Rect, app: &App) {
    let mut stats: Vec<_> = app.stats.iter().collect();
    
    // Sort based on current sort mode
    match app.sort_mode {
        SortMode::ByTime => stats.sort_by(|a, b| b.1.total_time.partial_cmp(&a.1.total_time).unwrap()),
        SortMode::ByName => stats.sort_by(|a, b| a.0.cmp(b.0)),
        SortMode::ByFrequency => stats.sort_by(|a, b| b.1.call_count.cmp(&a.1.call_count)),
    }

    let rows: Vec<Row> = stats
        .iter()
        .take(area.height as usize - 4)
        .map(|(name, stats)| {
            let bar_len = ((stats.total_time / 10.0).min(60.0)) as usize;
            let bar = "█".repeat(bar_len.max(1));
            
            let selected = if let Some(selected) = &app.selected_kernel {
                *name == selected // Fixed comparison
            } else {
                false
            };
            
            let style = if selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(name.as_str()).style(style), 
                Cell::from(format!("{:>8.3}", stats.total_time)),
                Cell::from(format!("{:>5}", stats.call_count)),
                Cell::from(format!("{:>8.3}", stats.avg_time)),
                Cell::from(bar).style(Style::default().fg(Color::Green)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(25),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Length(20),
        ],
    )
    .header(
        Row::new(vec!["Kernel", "Total(ms)", "Calls", "Avg(ms)", "Bar"])
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(" Kernel Statistics "))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn render_kernel_details_view(f: &mut Frame, area: Rect, app: &App) {
    if let Some(kernel_name) = &app.selected_kernel {
        let _kernel_records: Vec<_> = app.records.iter() 
            .filter(|r| r.kernel == *kernel_name)
            .collect();
            
        if let Some(stats) = app.stats.get(kernel_name) {
            let details = vec![
                Line::from(format!("Kernel: {}", kernel_name)),
                Line::from(""),
                Line::from(format!("Total Time: {:.3} ms", stats.total_time)),
                Line::from(format!("Call Count: {}", stats.call_count)),
                Line::from(format!("Average Time: {:.3} ms", stats.avg_time)),
                Line::from(format!("Min Time: {:.3} ms", stats.min_time)),
                Line::from(format!("Max Time: {:.3} ms", stats.max_time)),
            ];
            
            let paragraph = Paragraph::new(details)
                .block(Block::default().borders(Borders::ALL).title(" Kernel Details "))
                .wrap(Wrap { trim: true });
                
            f.render_widget(paragraph, area);
        } else {
            let paragraph = Paragraph::new("No statistics available for this kernel")
                .block(Block::default().borders(Borders::ALL).title(" Kernel Details "));
            f.render_widget(paragraph, area);
        }
    } else {
        let paragraph = Paragraph::new("No kernel selected")
            .block(Block::default().borders(Borders::ALL).title(" Kernel Details "));
        f.render_widget(paragraph, area);
    }
}

fn render_memory_view(f: &mut Frame, area: Rect, app: &App) {
    // Group records by kernel and sum memory usage
    let mut memory_by_kernel: HashMap<String, f64> = HashMap::new();
    
    for record in &app.records {
        if let Some(memory) = record.memory_usage {
            *memory_by_kernel.entry(record.kernel.clone()).or_insert(0.0) += memory;
        }
    }
    
    let mut memory_stats: Vec<_> = memory_by_kernel.iter().collect();
    memory_stats.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    let rows: Vec<Row> = memory_stats
        .iter()
        .take(area.height as usize - 4)
        .map(|(name, memory)| {
            let bar_len = ((**memory / 10.0).min(60.0)) as usize; 
            let bar = "█".repeat(bar_len.max(1));
            Row::new(vec![
                Cell::from(name.as_str()), 
                Cell::from(format!("{:>8.1}", **memory)), 
                Cell::from(bar).style(Style::default().fg(Color::Magenta)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(25),
            Constraint::Length(10),
            Constraint::Length(20),
        ],
    )
    .header(
        Row::new(vec!["Kernel", "Memory(MB)", "Bar"])
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(" Memory Usage "))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn render_footer(f: &mut Frame, area: Rect, app: &App) {
    let count = app.records.len();
    
    let footer_text = if count > 0 {
        let avg: f64 = app.records.iter().map(|r| r.time_ms).sum::<f64>() / count as f64;
        format!(
            " total entries: {:<5} | avg time: {:>8.3} ms | selected: {} ",
            count, 
            avg,
            app.selected_kernel.as_deref().unwrap_or("none")
        )
    } else {
        " No records available ".to_string()
    };
    
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(footer, area);
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

    let mut app = App::new();
    let mut last_update = std::time::Instant::now();

    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('r') => {
                        if fs::write(&log_path, "").is_ok() {
                            app.records.clear();
                            app.stats.clear();
                        }
                    }
                    KeyCode::Char('1') => app.view_mode = ViewMode::Timeline,
                    KeyCode::Char('2') => app.view_mode = ViewMode::Statistics,
                    KeyCode::Char('3') => app.view_mode = ViewMode::KernelDetails,
                    KeyCode::Char('4') => app.view_mode = ViewMode::MemoryUsage,
                    KeyCode::Char('s') => {
                        app.sort_mode = match app.sort_mode {
                            SortMode::ByTime => SortMode::ByName,
                            SortMode::ByName => SortMode::ByFrequency,
                            SortMode::ByFrequency => SortMode::ByTime,
                        };
                    }
                    KeyCode::Char('f') => {
                        app.filter_phase = match app.filter_phase.as_deref() {
                            None => Some("run".to_string()),
                            Some("run") => Some("compile".to_string()),
                            Some("compile") => Some("hit".to_string()),
                            _ => None,
                        };
                    }
                    KeyCode::Up => {
                        if let Some(selected) = &app.selected_kernel {
                            let kernels: Vec<_> = app.stats.keys().collect();
                            if let Some(pos) = kernels.iter().position(|k| *k == selected) { 
                                if pos > 0 {
                                    app.selected_kernel = Some(kernels[pos - 1].clone());
                                }
                            }
                        } else if !app.stats.is_empty() {
                            app.selected_kernel = app.stats.keys().next().cloned();
                        }
                    }
                    KeyCode::Down => {
                        if let Some(selected) = &app.selected_kernel {
                            let kernels: Vec<_> = app.stats.keys().collect();
                            if let Some(pos) = kernels.iter().position(|k| *k == selected) { 
                                if pos < kernels.len() - 1 {
                                    app.selected_kernel = Some(kernels[pos + 1].clone());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if last_update.elapsed() >= Duration::from_millis(500) {
            app.records = read_records(&log_path);
            app.update_stats();
            last_update = std::time::Instant::now();
        }

        terminal.draw(|f| {
            ui(f, &mut app);
        })?;

        sleep(Duration::from_millis(100));
    }

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}
