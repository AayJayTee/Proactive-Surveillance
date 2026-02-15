import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule],   // ✅ ADD THIS
  templateUrl: './results.html',
  styleUrl: './results.css'
})
export class ResultsComponent implements OnInit {

  report: any;

  constructor(private router: Router) {}

  ngOnInit(): void {
    const stored = localStorage.getItem('report');

    if (!stored) {
      this.router.navigate(['/']);
      return;
    }

    this.report = JSON.parse(stored);
  }
}
