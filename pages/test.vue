<template>
    <div class="flex container bg-red-700">
      <div class="timeline">
        <div class="timeline-item" v-for="(job, index) in jobs" :key="index">
          <div class="timeline-item-content">
            <h3 class="timeline-item-title">{{ job.title }}</h3>
            <h4 class="timeline-item-subtitle">{{ job.company }}</h4>
            <p class="timeline-item-date">{{ job.date }}</p>
            <p class="timeline-item-description">{{ job.description }}</p>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script lang="ts">
  export default {
    data() {
      return {
        jobs: [
          {
            title: "Software Developer",
            company: "Microsoft",
            date: "2019 - Present",
            description:
              "Developed and maintained software applications for Microsoft's cloud platform."
          },
          {
            title: "Web Developer",
            company: "Google",
            date: "2016 - 2019",
            description:
              "Developed and maintained web applications for Google's search engine."
          },
          {
            title: "Software Engineer",
            company: "Amazon",
            date: "2014 - 2016",
            description:
              "Developed and maintained software applications for Amazon's e-commerce platform."
          }
        ]
      };
    }
  };
  </script>
  
  <style>
  .timeline {
    position: relative;
    max-width: 1200px;
    margin: 0 auto;
  }
  
  .timeline::before {
    content: "";
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 2px;
    height: 100%;
    background-color: #ddd;
  }
  
  .timeline-item {
    position: relative;
    margin-bottom: 50px;
  }
  
  .timeline-item::before {
    content: "";
    position: absolute;
    top: -10px;
    left: -10px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background-color: #fff;
    border: 2px solid #ddd;
  }
  
  .timeline-item::after {
    content: "";
    position: absolute;
    top: calc(50% - 5px);
    left: -10px;
    width: 20px;
    height: 10px;
  }
  
  .timeline-item-content {
    position: relative;
  }
  
  .timeline-item-title {
      font-size : large; 
      font-weight : bold; 
  }
  
  .timeline-item-subtitle {
      font-size : medium; 
      font-weight : bold; 
  }
  
  .timeline-item-date {
      font-size : small; 
      font-style : italic; 
  }
  
  .timeline-item-description {
      font-size : medium; 
  }
  </style>