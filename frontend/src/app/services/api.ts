import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private baseUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  analyzeVideo(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('video', file);

    return this.http.post(`${this.baseUrl}/analyze`, formData);
  }
}
