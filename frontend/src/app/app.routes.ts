import { Routes } from '@angular/router';
import { UploadComponent } from './pages/upload/upload';
import { ResultsComponent } from './pages/results/results';

export const routes: Routes = [
  { path: '', component: UploadComponent },
  { path: 'results', component: ResultsComponent }
];
