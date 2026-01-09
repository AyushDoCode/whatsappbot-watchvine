/**
 * Google Apps Script for WatchVine Order Management
 * Deploy this as a Web App to save orders to Google Sheets
 * 
 * HOW TO DEPLOY:
 * 1. Open your Google Sheet
 * 2. Go to Extensions → Apps Script
 * 3. Delete default code and paste this entire script
 * 4. Click Deploy → New deployment
 * 5. Type: Web app
 * 6. Execute as: Me
 * 7. Who has access: Anyone
 * 8. Copy the Web App URL and secret key
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

// Your secret key (change this to a strong random string)
const SECRET_KEY = "your_secret_key_here_change_me_12345";

// Sheet name where orders will be saved
const SHEET_NAME = "Orders";

// ============================================================================
// MAIN FUNCTION - Receives POST requests
// ============================================================================

function doPost(e) {
  try {
    // Parse incoming JSON data
    const data = JSON.parse(e.postData.contents);
    
    // Verify secret key
    if (data.secret !== SECRET_KEY) {
      return createResponse(false, "Invalid secret key");
    }
    
    // Get order data
    const orderData = data.order;
    
    // Save to sheet
    const result = saveOrderToSheet(orderData);
    
    if (result.success) {
      return createResponse(true, "Order saved successfully", result.rowNumber);
    } else {
      return createResponse(false, result.error);
    }
    
  } catch (error) {
    return createResponse(false, "Error: " + error.toString());
  }
}

// ============================================================================
// SAVE ORDER TO GOOGLE SHEET
// ============================================================================

function saveOrderToSheet(orderData) {
  try {
    // Get active spreadsheet
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    
    // Get or create Orders sheet
    let sheet = ss.getSheetByName(SHEET_NAME);
    if (!sheet) {
      sheet = ss.insertSheet(SHEET_NAME);
      // Add headers
      const headers = [
        "Order ID",
        "Timestamp",
        "Customer Name",
        "Phone Number",
        "Email",
        "Address",
        "Product Name",
        "Product URL",
        "Quantity",
        "Status",
        "Notes"
      ];
      sheet.appendRow(headers);
      
      // Format header row
      const headerRange = sheet.getRange(1, 1, 1, headers.length);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#4285f4");
      headerRange.setFontColor("#ffffff");
    }
    
    // Prepare row data
    const row = [
      orderData.order_id || "",
      orderData.timestamp || new Date().toLocaleString(),
      orderData.customer_name || "",
      orderData.phone_number || "",
      orderData.email || "N/A",
      orderData.address || "",
      orderData.product_name || "",
      orderData.product_url || "",
      orderData.quantity || 1,
      orderData.status || "Pending",
      orderData.notes || ""
    ];
    
    // Append row to sheet
    sheet.appendRow(row);
    
    // Get row number
    const lastRow = sheet.getLastRow();
    
    // Auto-resize columns for better visibility
    sheet.autoResizeColumns(1, row.length);
    
    return {
      success: true,
      rowNumber: lastRow
    };
    
  } catch (error) {
    return {
      success: false,
      error: error.toString()
    };
  }
}

// ============================================================================
// TEST FUNCTION - For testing GET requests
// ============================================================================

function doGet(e) {
  return ContentService.createTextOutput(
    JSON.stringify({
      status: "success",
      message: "WatchVine Order Management API is running!",
      timestamp: new Date().toISOString(),
      info: "Send POST requests with order data to save orders"
    })
  ).setMimeType(ContentService.MimeType.JSON);
}

// ============================================================================
// HELPER FUNCTION - Create JSON response
// ============================================================================

function createResponse(success, message, data = null) {
  const response = {
    success: success,
    message: message,
    timestamp: new Date().toISOString()
  };
  
  if (data !== null) {
    response.data = data;
  }
  
  return ContentService.createTextOutput(JSON.stringify(response))
    .setMimeType(ContentService.MimeType.JSON);
}

// ============================================================================
// TEST FUNCTION - Manual test (Run this to test)
// ============================================================================

function testSaveOrder() {
  const testOrder = {
    order_id: "TEST_" + new Date().getTime(),
    timestamp: new Date().toLocaleString(),
    customer_name: "Test Customer",
    phone_number: "9999999999",
    email: "test@example.com",
    address: "Test Address, Test City",
    product_name: "Test Product",
    product_url: "https://example.com/product",
    quantity: 1,
    status: "Pending",
    notes: "Test order"
  };
  
  const result = saveOrderToSheet(testOrder);
  Logger.log(result);
  
  if (result.success) {
    Logger.log("✅ Test order saved successfully at row: " + result.rowNumber);
  } else {
    Logger.log("❌ Error: " + result.error);
  }
}
