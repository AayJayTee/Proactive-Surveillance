import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api';
import { Router } from '@angular/router';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],   // ✅ ADD THIS
  templateUrl: './upload.html',
  styleUrl: './upload.css'
})
export class UploadComponent {

  selectedFile!: File;
  loading = false;

  constructor(private api: ApiService, private router: Router) {}

  onFileChange(event: any) {
    this.selectedFile = event.target.files[0];
  }

  onUpload() {
    if (!this.selectedFile) return;

    this.loading = true;

    this.api.analyzeVideo(this.selectedFile)
      .subscribe(response => {
        localStorage.setItem('report', JSON.stringify(response));
        this.loading = false;
        this.router.navigate(['/results']);
      }, error => {
        console.error(error);
        this.loading = false;
      });
  }
}
